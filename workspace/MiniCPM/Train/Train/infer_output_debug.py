#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MiniCPM-O-4.5 推理调试脚本 (infer_debug.py) —— v17-fix 全双工连续 SPEAK 修复版
放置于 Train 目录，与 MiniCPMO45 同级。

修复：
  - 修复 f-string 嵌套单引号导致的 SyntaxError
  - 全双工：音频结束后，若模型未 turn_eos，继续用静音 chunk 推进
  - 全双工：增加 max_new_speak_tokens_per_chunk 参数（默认 50）
  - 全双工：打印跨 chunk 连续 SPEAK 拼接效果
"""

import argparse
import os
import sys
import json
from copy import deepcopy
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import librosa
import soundfile as sf

from transformers import AutoConfig
from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO, ProcessorMode
from MiniCPMO45.utils import TTSSamplingParams


# ==================== 打印工具 ====================

def print_section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_tokens(title: str, tokenizer, ids, skip_special: bool = False, limit: int = 100):
    if isinstance(ids, torch.Tensor):
        ids = ids.cpu().tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = list(ids)
    text = tokenizer.decode(ids, skip_special_tokens=skip_special)
    print(f"\n>>> {title}  (len={len(ids)})")
    print(f"    IDs: {ids[:limit]}{' ...' if len(ids) > limit else ''}")
    print(f"    Decode: {repr(text[:500])}{' ...' if len(text) > 500 else ''}")


# ==================== Schema 构建工具（半双工） ====================

def build_schema_items(tokenizer, input_ids, audio_bounds=None, image_bound=None):
    if isinstance(input_ids, torch.Tensor):
        ids = input_ids[0].cpu().tolist()
    elif isinstance(input_ids, (list, tuple)):
        ids = list(input_ids[0]) if isinstance(input_ids[0], (list, tuple)) else list(input_ids)
    else:
        ids = list(input_ids)

    ranges = []

    def _flatten_bounds(obj):
        if isinstance(obj, (list, tuple)) and len(obj) == 2 and isinstance(obj[0], int):
            yield obj
        elif isinstance(obj, (list, tuple)):
            for sub in obj:
                yield from _flatten_bounds(sub)

    if audio_bounds is not None:
        ab = audio_bounds[0] if isinstance(audio_bounds, (list, tuple)) else audio_bounds
        if isinstance(ab, torch.Tensor):
            ab = ab.tolist()
        for b in _flatten_bounds(ab):
            s, e = int(b[0]), int(b[1])
            if 0 <= s < e <= len(ids):
                ranges.append((s, e, "audio", e - s))

    if image_bound is not None:
        ib = image_bound[0] if isinstance(image_bound, (list, tuple)) else image_bound
        if isinstance(ib, torch.Tensor):
            ib = ib.tolist()
        for b in _flatten_bounds(ib):
            s, e = int(b[0]), int(b[1])
            if 0 <= s < e <= len(ids):
                ranges.append((s, e, "img", e - s))

    ranges.sort(key=lambda x: x[0])

    items = []
    i = 0
    while i < len(ids):
        matched = False
        for s, e, rtype, rlen in ranges:
            if s <= i < e:
                items.append((rtype, rlen))
                i = e
                matched = True
                break
        if not matched:
            items.append(ids[i])
            i += 1
    return items


def decode_schema_items(tokenizer, items, include_embeddings=True):
    parts = []
    for item in items:
        if isinstance(item, tuple):
            etype, elen = item
            if include_embeddings:
                parts.append(f"[{etype}_embed_{elen}]")
        else:
            parts.append(tokenizer.decode([int(item)], skip_special_tokens=False))
    return "".join(parts)


def save_intermediate(data: dict, filename: str):
    path = os.path.join(os.path.dirname(__file__), filename)

    def _serialize(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialize(v) for v in obj]
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_serialize(data), f, ensure_ascii=False, indent=2)
    print(f"\n[保存] 中间结果 -> {path}")


# ==================== 全双工专用解码工具 ====================

def decode_full_duplex_input(tokenizer, items):
    if not items:
        return ""

    special_map = {
        tokenizer.convert_tokens_to_ids("<unit>"): "[UNIT]",
        tokenizer.convert_tokens_to_ids("</unit>"): "[/UNIT]",
        tokenizer.convert_tokens_to_ids("<image>"): "[IMAGE]",
        tokenizer.convert_tokens_to_ids("</image>"): "[/IMAGE]",
        tokenizer.convert_tokens_to_ids("<slice>"): "[SLICE]",
        tokenizer.convert_tokens_to_ids("</slice>"): "[/SLICE]",
    }

    parts = []
    for item in items:
        if isinstance(item, tuple):
            etype, elen = item
            parts.append(f"[{etype}_embed_{elen}]")
        else:
            tid = int(item)
            if tid in special_map:
                parts.append(special_map[tid])
            else:
                parts.append(tokenizer.decode([tid], skip_special_tokens=False))
    return "".join(parts)


def decode_full_duplex_output(tokenizer, ids):
    if not ids:
        return ""

    raw_text = tokenizer.decode(ids, skip_special_tokens=False)

    highlights = {
        "||<|<|listen|>": "[LISTEN]",
        "||<|<|speak|>": "[SPEAK]",
        "||<|<|chunk_eos|>": "[CHUNK_EOS]",
        "||<|<|turn_eos|>": "[TURN_EOS]",
        "||<|<|tts_bos|>": "[TTS_BOS]",
        "||<|<|tts_eos|>": "[TTS_EOS]",
        "||<|<|tts_pad|>": "[TTS_PAD]",
        "</unit>": "[/UNIT]",
    }
    for token_str, highlight in highlights.items():
        raw_text = raw_text.replace(token_str, highlight)

    return raw_text


# ==================== 半双工模式 ====================

def run_half_duplex(model, tokenizer, msgs, audio_input, tts_ref_audio, args):
    print_section("【半双工模式】输入处理 & 主干模型生成 & 语音 Token 分析")

    copy_msgs = deepcopy(msgs)
    images = []
    audios = []
    audio_parts = []
    for i, msg in enumerate(copy_msgs):
        content = msg["content"]
        if isinstance(content, str):
            content = [content]
        cur_msgs = []
        for c in content:
            if isinstance(c, np.ndarray):
                audios.append(c)
                audio_parts.append(i)
                cur_msgs.append("<audio>./</audio>")
            elif isinstance(c, str):
                cur_msgs.append(c)
        msg["content"] = "\n".join(cur_msgs)

    prompt = tokenizer.apply_chat_template(
        copy_msgs,
        tokenize=False,
        add_generation_prompt=True,
        use_tts_template=True,
    )
    print(f">>> 半双工 Prompt 字符串:\n{prompt}")

    inputs = model.processor(
        [prompt],
        [images],
        [audios],
        [audio_parts] if audio_parts else None,
        max_slice_nums=None,
        use_image_id=None,
        stream_input=False,
        return_tensors="pt",
        max_length=8192,
    ).to(model.device)

    for key in ["tgt_sizes", "pixel_values", "image_bound", "spk_bounds",
                "audio_bounds", "audio_features", "audio_feature_lens"]:
        if key not in inputs:
            inputs[key] = None

    input_ids = inputs["input_ids"]
    audio_bounds = inputs.get("audio_bounds")
    image_bound = inputs.get("image_bound")

    input_schema_items = build_schema_items(tokenizer, input_ids, audio_bounds, image_bound)
    input_schema_text = decode_schema_items(tokenizer, input_schema_items, include_embeddings=True)

    print("\n>>> 半双工输入 Token Schema（文本占位符形式，完整）：")
    print(input_schema_text)
    print("\n>>> 半双工输入 Token Schema（结构化 items）：")
    print(json.dumps([str(x) for x in input_schema_items], ensure_ascii=False, indent=2))

    _orig_decode = model._decode
    captured_outputs = {}

    def _hooked_decode(inputs_embeds, tokenizer, attention_mask, **kwargs):
        outputs = _orig_decode(inputs_embeds, tokenizer, attention_mask, **kwargs)
        captured_outputs["outputs"] = outputs
        return outputs

    model._decode = _hooked_decode

    _orig_tts_generate = model.tts.generate
    captured_tts_info = {}

    def _hooked_tts_generate(inputs_embeds, eos_token, force_no_stop=False,
                             min_new_token=50, max_new_token=2048,
                             show_tqdm=True, streaming=False,
                             text_lengths=None,
                             sampling_params=TTSSamplingParams()):
        print_section("Hook: TTS.generate (VQ Token 生成)")
        print(f"    TTS inputs_embeds shape: {tuple(inputs_embeds.shape)}")
        print(f"    sampling_params: temp={sampling_params.temperature}, "
              f"top_p={sampling_params.top_p}, top_k={sampling_params.top_k}")

        result = _orig_tts_generate(
            inputs_embeds=inputs_embeds, eos_token=eos_token,
            force_no_stop=force_no_stop, min_new_token=min_new_token,
            max_new_token=max_new_token, show_tqdm=show_tqdm,
            streaming=streaming, text_lengths=text_lengths,
            sampling_params=sampling_params,
        )

        print_section("Hook: TTS.generate 出口")
        if result.new_ids is not None:
            print(f"    VQ new_ids shape: {tuple(result.new_ids.shape)}")
            if result.new_ids.dim() == 3:
                vq_tokens_1st = result.new_ids[0, :, 0].cpu().tolist()
                print(f">>> VQ Token 第1码本序列 (len={len(vq_tokens_1st)})")
                print(f"    前50: {vq_tokens_1st[:50]}")
                print(f"    后50: {vq_tokens_1st[-50:]}")
                captured_tts_info["vq_tokens_1st"] = vq_tokens_1st
                captured_tts_info["vq_shape"] = list(result.new_ids.shape)
                captured_tts_info["finished"] = result.finished
        return result

    model.tts.generate = _hooked_tts_generate

    try:
        tts_sampling_params = TTSSamplingParams(
            temperature=args.tts_temperature,
            top_p=args.tts_top_p,
            top_k=args.tts_top_k,
        )
        result = model.chat(
            msgs=msgs,
            generate_audio=args.generate_audio,
            use_tts_template=True,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            tts_ref_audio=tts_ref_audio,
            tts_sampling_params=tts_sampling_params,
        )
    finally:
        model._decode = _orig_decode
        model.tts.generate = _orig_tts_generate

    outputs = captured_outputs.get("outputs")
    if outputs is not None and hasattr(outputs, "sequences"):
        generated_ids = outputs.sequences[0]
        print_tokens("半双工主干 LLM 输出 Token IDs", tokenizer, generated_ids)
        output_schema_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        print("\n>>> 半双工主干 LLM 输出 Token Schema（文本占位符形式，完整）：")
        print(output_schema_text)

        full_seq = torch.cat([input_ids[0].cpu(), generated_ids.cpu()])
        full_seq_list = full_seq.tolist()

        tts_bos_id = tokenizer.convert_tokens_to_ids("||<|<|tts_bos|>")
        tts_eos_id = tokenizer.convert_tokens_to_ids("||<|<|tts_eos|>")
        tts_bos_positions = [i for i, x in enumerate(full_seq_list) if x == tts_bos_id]
        tts_eos_positions = [i for i, x in enumerate(full_seq_list) if x == tts_eos_id]

        print(f"\n>>> 语音相关特殊 Token 位置分析：")
        print(f"    <|tts_bos|> (id={tts_bos_id}) 出现在位置: {tts_bos_positions}")
        print(f"    <|tts_eos|> (id={tts_eos_id}) 出现在位置: {tts_eos_positions}")

        if tts_bos_positions and tts_eos_positions:
            last_bos = tts_bos_positions[-1]
            last_eos = tts_eos_positions[-1]
            if last_bos < last_eos:
                tts_text_tokens = full_seq_list[last_bos:last_eos]
                print(f"\n>>> TTS 文本条件 Token (位置 {last_bos}~{last_eos})：")
                print_tokens("TTS 文本条件", tokenizer, tts_text_tokens, skip_special=False)

        if captured_tts_info:
            print(f"\n>>> VQ Token 分析：")
            print(f"    VQ 形状: {captured_tts_info.get('vq_shape')}")
            print(f"    是否完成: {captured_tts_info.get('finished')}")
            vq_1st = captured_tts_info.get("vq_tokens_1st", [])
            print(f"    第1码本序列长度: {len(vq_1st)}")
            if len(vq_1st) > 0:
                print(f"    前30: {vq_1st[:30]}")
                print(f"    后30: {vq_1st[-30:]}")

        intermediate = {
            "mode": "half_duplex",
            "prompt": prompt,
            "input_schema_items": [str(x) for x in input_schema_items],
            "input_schema_text": input_schema_text,
            "input_ids": input_ids[0].cpu().tolist(),
            "audio_bounds": audio_bounds[0].tolist() if audio_bounds is not None else None,
            "image_bound": image_bound[0].tolist() if image_bound is not None else None,
            "output_ids": generated_ids.cpu().tolist(),
            "output_schema_text": output_schema_text,
            "full_sequence_ids": full_seq_list,
            "full_sequence_text": tokenizer.decode(full_seq, skip_special_tokens=False),
            "tts_bos_positions": tts_bos_positions,
            "tts_eos_positions": tts_eos_positions,
            "vq_tokens_info": {
                "shape": captured_tts_info.get("vq_shape"),
                "finished": captured_tts_info.get("finished"),
                "vq_1st_sample": captured_tts_info.get("vq_tokens_1st", [])[:100],
            },
        }
        save_intermediate(intermediate, "half_duplex_intermediate.json")
    else:
        print("[警告] 未能捕获 LLM 输出")
        output_schema_text = ""
        generated_ids = torch.tensor([], dtype=torch.long)

    if isinstance(result, tuple):
        text, waveform = result
    else:
        text = result
        waveform = None

    return text, waveform


# ==================== 全双工模式（修复：连续 SPEAK + 静音推进） ====================

def run_full_duplex(model, tokenizer, audio_input, args):
    print_section("【全双工模式】处理流程（逐 chunk 输入，连续 SPEAK 直到 turn_eos）")

    model.set_mode(ProcessorMode.DUPLEX)

    ref_audio = audio_input
    prompt_wav_path = args.ref_audio or args.input_audio

    prefix = "||<|<|im_start|>system\nYou are a helpful assistant.\n<<||<|<|audio_start|>"
    suffix = "||<|<|audio_end|>哦"

    full_prompt = model.duplex_prepare(
        prefix_system_prompt=prefix,
        suffix_system_prompt=suffix,
        ref_audio=ref_audio,
        prompt_wav_path=prompt_wav_path,
    )
    print(f">>> 全双工 System Prompt:\n{full_prompt}")

    chunk_size = int(1.0 * 16000)  # 1s @ 16kHz
    total_len = len(audio_input)
    num_real_chunks = (total_len + chunk_size - 1) // chunk_size
    print(f">>> 音频总长度: {total_len} samples ({total_len/16000:.1f}s), 共 {num_real_chunks} 个真实 chunk")

    unit_records = []
    all_speak_texts = []
    all_audio_chunks = []

    # 1. 处理所有真实音频 chunk
    for i in range(num_real_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_len)
        chunk = audio_input[start:end]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode="constant")

        _process_chunk(model, tokenizer, i, chunk, start, end, unit_records,
                       all_speak_texts, all_audio_chunks, is_silence=False)

    # 2. 修复：音频结束后，如果模型还在 SPEAK（未 turn_eos），继续用静音推进
    max_silence_chunks = 10
    silence_chunk = np.zeros(chunk_size, dtype=np.float32)

    last_chunk_was_speak = (unit_records and not unit_records[-1]['is_listen']
                            and not unit_records[-1]['end_of_turn'])

    if last_chunk_was_speak:
        print(f"\n{'='*80}")
        print(f">>> 音频已结束，但模型仍在 SPEAK（未 turn_eos），进入静音推进模式")
        print(f">>> 最多推进 {max_silence_chunks} 个静音 chunk，直到模型 turn_eos 或 LISTEN")
        print(f"{'='*80}")

    silence_idx = 0
    while last_chunk_was_speak and silence_idx < max_silence_chunks:
        silence_idx += 1
        _process_chunk(model, tokenizer, num_real_chunks + silence_idx - 1,
                       silence_chunk, 0, 0, unit_records,
                       all_speak_texts, all_audio_chunks, is_silence=True)

        if unit_records[-1]['end_of_turn'] or unit_records[-1]['is_listen']:
            print(f"\n>>> 静音推进结束：模型已 {'turn_eos' if unit_records[-1]['end_of_turn'] else 'LISTEN'}")
            break

    if silence_idx == max_silence_chunks and last_chunk_was_speak:
        print(f"\n>>> [警告] 达到最大静音推进次数 ({max_silence_chunks})，模型仍未 turn_eos，强制结束")

    # 3. 累积完整会话视图
    print_section("【全双工模式】累积完整会话视图（所有 Unit 历史拼接）")
    session_schema = model.duplex.get_session_schema(include_embeddings=True)
    print(">>> 完整会话 Schema（所有 unit 拼接，含 KV cache 历史）：")
    print(session_schema)

    unit_count_open = session_schema.count("<unit>") + session_schema.count("[UNIT]")
    unit_count_close = session_schema.count("</unit>") + session_schema.count("[/UNIT]")
    print(f"\n>>> 会话级结构验证：")
    print(f"    <unit> / [UNIT] 总数: {unit_count_open}")
    print(f"    </unit> / [/UNIT] 总数: {unit_count_close}")
    if unit_count_open == unit_count_close:
        print("    ✓ 所有 Unit 均已闭合")
    else:
        print(f"    ✗ 闭合不匹配！缺少 {unit_count_open - unit_count_close} 个 </unit>")

    # 4. 汇总（展示跨 chunk 连续 SPEAK）
    print_section("【全双工模式】所有 Chunk 输入输出汇总（含静音推进）")
    speak_segments = []
    current_speak = []

    for rec in unit_records:
        # 修复 SyntaxError：避免 f-string 嵌套单引号，提前提取标签
        if rec.get('is_silence'):
            time_label = "静音"
            silence_label = "[SILENCE]"
        else:
            time_label = f"{rec['audio_start_s']:.1f}s~{rec['audio_end_s']:.1f}s"
            silence_label = ""

        print(f"\n--- Chunk {rec['chunk_idx']} ({time_label} {silence_label}) ---")
        print(f"Prefill:  {rec['prefill_schema'][:150]}{'...' if len(rec['prefill_schema'])>150 else ''}")
        print(f"Generate: {rec['generate_output'][:150]}{'...' if len(rec['generate_output'])>150 else ''}")
        print(f"Full:     {rec['full_unit_schema'][:200]}{'...' if len(rec['full_unit_schema'])>200 else ''}")
        print(f"模型文本: {rec['model_text']}")
        print(f"决策: is_listen={rec['is_listen']}, end_of_turn={rec['end_of_turn']}")

        # 收集连续 SPEAK 片段
        if not rec['is_listen'] and rec['model_text']:
            current_speak.append(rec['model_text'])
        else:
            if current_speak:
                speak_segments.append("".join(current_speak))
                current_speak = []

    if current_speak:
        speak_segments.append("".join(current_speak))

    print(f"\n>>> 跨 chunk 连续 SPEAK 片段（共 {len(speak_segments)} 段）：")
    for idx, seg in enumerate(speak_segments):
        print(f"    段落 {idx}: '{seg}'")

    # 5. 最终输出拼接与保存
    print_section("【全双工模式】最终输出拼接与保存")

    full_text = "".join(all_speak_texts)
    print(f">>> 全双工完整对话文本（所有 SPEAK 片段拼接）：")
    print(full_text)

    with open("full_duplex_output.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"\n[保存] 全双工文本 -> full_duplex_output.txt")

    if all_audio_chunks:
        full_waveform = np.concatenate(all_audio_chunks)
        sf.write("full_duplex_output.wav", full_waveform, samplerate=24000)
        print(f"[保存] 全双工音频 -> full_duplex_output.wav "
              f"({len(full_waveform)/24000:.2f}s @ 24kHz, {len(all_audio_chunks)} 个片段)")
    else:
        print("[保存] 全双工无音频输出")

    # 6. 保存中间结果
    intermediate = {
        "mode": "full_duplex",
        "system_prompt": full_prompt,
        "unit_records": unit_records,
        "session_schema": session_schema,
        "full_text": full_text,
        "speak_segments": speak_segments,
        "num_audio_chunks": len(all_audio_chunks),
        "num_real_chunks": num_real_chunks,
        "num_silence_chunks": silence_idx,
        "prefill_schema_tokens": [
            [str(x) for x in unit] for unit in model.duplex.prefill_schema_tokens
        ],
        "total_ids": model.duplex.total_ids,
        "total_ids_text": tokenizer.decode(model.duplex.total_ids, skip_special_tokens=False) if model.duplex.total_ids else "",
    }
    save_intermediate(intermediate, "full_duplex_intermediate.json")

    return session_schema, full_text, all_audio_chunks


def _process_chunk(model, tokenizer, chunk_idx, chunk, start, end, unit_records,
                   all_speak_texts, all_audio_chunks, is_silence=False):
    """处理单个 chunk（真实音频或静音），打印输入输出，收集结果"""
    label = "SILENCE" if is_silence else f"{start/16000:.1f}s~{end/16000:.1f}s"
    print(f"\n{'='*60}")
    print(f">>> 全双工 Chunk {chunk_idx} [{label}] {'[静音推进]' if is_silence else ''}")
    print(f"{'='*60}")

    # 1. Prefill
    prefill_res = model.duplex_prefill(audio_waveform=chunk, frame_list=None, max_slice_nums=1)
    print(f"    [Prefill] cost={prefill_res.get('cost_all', 0):.3f}s")

    if model.duplex.prefill_schema_tokens and len(model.duplex.prefill_schema_tokens) > chunk_idx:
        current_prefill = model.duplex.prefill_schema_tokens[chunk_idx]
        input_schema_text = decode_full_duplex_input(tokenizer, current_prefill)
        print(f"\n>>> Chunk {chunk_idx} Prefill 输入（占位符形式）：")
        print(input_schema_text)
    else:
        input_schema_text = ""

    gen_start_len = len(model.duplex.total_ids)

    # 2. Generate
    gen_res = model.duplex_generate(
        decode_mode="sampling",
        temperature=0.7,
        top_k=20,
        top_p=0.8,
        listen_prob_scale=1.0,
        listen_top_k=5,
        text_repetition_penalty=1.05,
        text_repetition_window_size=512,
        length_penalty=1.1,
        force_listen_override=False,
    )

    gen_end_len = len(model.duplex.total_ids)
    current_gen_ids = model.duplex.total_ids[gen_start_len:gen_end_len]

    print(f"\n>>> Chunk {chunk_idx} LLM 生成结果：")
    print(f"    is_listen={gen_res['is_listen']}")
    print(f"    end_of_turn={gen_res['end_of_turn']}")
    print(f"    模型原生 text='{gen_res['text']}'")

    # 收集音频和文本
    if not gen_res['is_listen'] and gen_res.get('audio_waveform') is not None:
        waveform = gen_res['audio_waveform']
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        all_audio_chunks.append(waveform)
        if gen_res['text']:
            all_speak_texts.append(gen_res['text'])

    if current_gen_ids:
        output_schema_text = decode_full_duplex_output(tokenizer, current_gen_ids)
        print(f"\n>>> Chunk {chunk_idx} Generate 输出（占位符形式）：")
        print(output_schema_text)
        print(f"\n>>> 输出 Token IDs (len={len(current_gen_ids)})：")
        print(current_gen_ids)

        listen_id = tokenizer.convert_tokens_to_ids("||<|<|listen|>")
        speak_id = tokenizer.convert_tokens_to_ids("||<|<|speak|>")
        chunk_eos_id = tokenizer.convert_tokens_to_ids("||<|<|chunk_eos|>")
        turn_eos_id = tokenizer.convert_tokens_to_ids("||<|<|turn_eos|>")

        has_listen = listen_id in current_gen_ids
        has_speak = speak_id in current_gen_ids
        has_chunk_eos = chunk_eos_id in current_gen_ids
        has_turn_eos = turn_eos_id in current_gen_ids

        print(f"\n>>> 输出结构分析：")
        print(f"    [LISTEN]: {has_listen} | [SPEAK]: {has_speak} | "
              f"[CHUNK_EOS]: {has_chunk_eos} | [TURN_EOS]: {has_turn_eos}")
    else:
        output_schema_text = ""
        print("\n>>> [警告] 当前 Chunk 无 LLM 输出 token")

    # 3. Finalize
    model.duplex_finalize()

    # 4. 打印完整闭合结构
    current_unit_schemas = model.duplex.get_unit_schemas(include_embeddings=True)
    if len(current_unit_schemas) > chunk_idx:
        full_unit_schema = current_unit_schemas[chunk_idx]
        print(f"\n>>> Chunk {chunk_idx} 完整闭合 Unit Schema：")
        print(full_unit_schema)

        has_open = "<unit>" in full_unit_schema or "[UNIT]" in full_unit_schema
        has_close = "</unit>" in full_unit_schema or "[/UNIT]" in full_unit_schema
        status = "✓ 闭合" if (has_open and has_close) else "✗ 未闭合"
        print(f"    结构验证: {status}")
    else:
        full_unit_schema = ""

    unit_records.append({
        "chunk_idx": chunk_idx,
        "audio_start_s": start / 16000.0 if not is_silence else 0,
        "audio_end_s": end / 16000.0 if not is_silence else 0,
        "is_silence": is_silence,
        "prefill_schema": input_schema_text,
        "generate_output": output_schema_text,
        "full_unit_schema": full_unit_schema,
        "model_text": gen_res["text"],
        "is_listen": gen_res["is_listen"],
        "end_of_turn": gen_res["end_of_turn"],
    })


# ==================== CLI 入口 ====================

def parse_args():
    parser = argparse.ArgumentParser(description="MiniCPM-O-4.5 半双工/全双工 Token 对比调试脚本")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--input-audio", type=str, required=True)
    parser.add_argument("--input-text", type=str, default="")
    parser.add_argument("--ref-audio", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--tts-temperature", type=float, default=0.8)
    parser.add_argument("--tts-top-p", type=float, default=0.85)
    parser.add_argument("--tts-top-k", type=int, default=25)
    parser.add_argument("--max-speak-tokens", type=int, default=50,
                        help="全双工每 chunk 最大 SPEAK token 数（默认 50，原版 20 容易截断）")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    args.generate_audio = True

    print(f"[初始化] 读取并修补 config: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    if hasattr(config, 'tts_config') and config.tts_config is not None:
        for attr, default in [('top_p', 0.85), ('top_k', 25), ('repetition_penalty', 1.05)]:
            if not hasattr(config.tts_config, attr):
                setattr(config.tts_config, attr, default)
                print(f"    [修复] config.tts_config.{attr} = {default}")

    print(f"[初始化] 加载模型 (dtype={args.dtype}, device={args.device}) ...")
    model = MiniCPMO.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device,
    )
    model.eval()

    print("[初始化] Unified 初始化 (chat_vocoder=token2wav) ...")
    duplex_cfg = {
        "max_new_speak_tokens_per_chunk": args.max_speak_tokens,
    }
    model.init_unified(pt_path=None, chat_vocoder="token2wav", preload_both_tts=False, duplex_config=duplex_cfg)
    print(f"[初始化] 完成，max_new_speak_tokens_per_chunk={args.max_speak_tokens}")

    tokenizer = model.processor.tokenizer

    audio_input, sr = librosa.load(args.input_audio, sr=16000, mono=True)
    print(f"[输入] 加载音频: {args.input_audio}, 时长={len(audio_input) / 16000:.2f}s")

    tts_ref_audio = None
    if args.ref_audio:
        tts_ref_audio, _ = librosa.load(args.ref_audio, sr=16000, mono=True)
    else:
        tts_ref_audio = audio_input

    user_content = [audio_input]
    if args.input_text:
        user_content.append(args.input_text)

    msgs = [
        {"role": "system", "content": [args.system_prompt]},
        {"role": "user", "content": user_content},
    ]

    # 4. 运行半双工
    hd_text, hd_waveform = run_half_duplex(model, tokenizer, msgs, audio_input, tts_ref_audio, args)

    with open("half_duplex_output.txt", "w", encoding="utf-8") as f:
        f.write(hd_text)
    print(f"\n[保存] 半双工文本 -> half_duplex_output.txt")
    print(f"内容:\n{hd_text}")

    if hd_waveform is not None:
        sf.write("half_duplex_output.wav", hd_waveform, samplerate=24000)
        print(f"[保存] 半双工音频 -> half_duplex_output.wav ({len(hd_waveform) / 24000:.2f}s @ 24kHz)")
    else:
        print("[保存] 半双工无音频输出")

    # 5. 运行全双工
    fd_schema, fd_text, fd_audio = run_full_duplex(model, tokenizer, audio_input, args)

    # 6. 对比总结
    print_section("【对比总结】半双工 vs 全双工")
    print("【输入侧】")
    print("  半双工 (Half-Duplex / Chat):")
    print("    - 标准 Chat Template：")
    print("      <|im_start|>system ... 哦 <|im_start|>user ... 哦 <|im_start|>assistant ...")
    print("    - Audio 占位符: [audio_embed_50]")
    print("    - Image 占位符: [img_embed_64]")
    print("    - 无 <unit> 包裹，无帧级切分")
    print()
    print("  全双工 (Full-Duplex / Duplex):")
    print("    - 逐 chunk（1秒/帧）输入，每帧包裹在 <unit> ... </unit> 中")
    print("    - 音频结束后，若模型仍在 SPEAK，用静音 chunk 推进直到 turn_eos")
    print()
    print("【输出侧（主干 LLM）】")
    print("  半双工:")
    print("    - 标准文本 token + 特殊标记：")
    print("      你好呀！<<||<|<|tts_bos|> ... <|tts_eos|> 哦")
    print("    - 无决策 token，无帧级终止符")
    print()
    print("  全双工:")
    print("    - 每帧最多生成 N 个 token（--max-speak-tokens 控制，默认 50）")
    print("    - 若 N 个 token 没说完：发 [CHUNK_EOS]，下 chunk 继续 SPEAK")
    print("    - 若说完了：发 [TURN_EOS]，下 chunk 转为 LISTEN")
    print("    - 音频结束后，用静音维持，让模型把话说完")
    print()
    print("【跨 chunk 连续 SPEAK 示例】")
    print("  Chunk 3: [SPEAK]你好，有[CHUNK_EOS]  → 话没说完")
    print("  Chunk 4: [SPEAK]什么可以帮您的吗？[TURN_EOS]  → 说完了")
    print()
    print("【最终输出文件】")
    print("  半双工: half_duplex_output.txt / .wav / intermediate.json")
    print("  全双工: full_duplex_output.txt / .wav / intermediate.json")


if __name__ == "__main__":
    main()
