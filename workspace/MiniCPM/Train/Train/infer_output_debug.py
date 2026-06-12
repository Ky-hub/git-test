#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MiniCPM-O-4.5 推理调试脚本 (infer_debug.py) —— v11 双模式对比修复版
放置于 Train 目录，与 MiniCPMO45 同级。

修复：
  - 半双工：Hook TTS.generate 打印 VQ Token 序列，完整分析语音链路
  - 全双工：修复 duplex_generate 参数透传 None 导致的 TypeError
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


def print_tensor_info(title: str, tensor: torch.Tensor):
    print(f"    {title}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}")


# ==================== Schema 构建工具 ====================

def build_schema_items(tokenizer, input_ids, audio_bounds=None, image_bound=None):
    """
    将 input_ids 与 audio_bounds / image_bound 交叉，
    生成混合列表：int (正常 token_id) 或 tuple ("audio"|"img", length)。
    """
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
    """将 schema_items 解码为可读文本（占位符形式）。"""
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
    """保存中间结果到本地 JSON（支持人工修改后重载）。"""
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


# ==================== 半双工模式（完整输入输出 + 语音 Token 分析） ====================

def run_half_duplex(model, tokenizer, msgs, audio_input, tts_ref_audio, args):
    print_section("【半双工模式】输入处理 & 主干模型生成 & 语音 Token 分析")

    # ---------- 1. 独立复现 chat() 的消息解析，用于打印输入结构 ----------
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

    # ---------- 2. Processor 预处理（仅用于分析输入结构） ----------
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

    # ---------- 3. 输入 Token Schema（占位符形式） ----------
    input_schema_items = build_schema_items(tokenizer, input_ids, audio_bounds, image_bound)
    input_schema_text = decode_schema_items(tokenizer, input_schema_items, include_embeddings=True)

    print("\n>>> 半双工输入 Token Schema（文本占位符形式，完整）：")
    print(input_schema_text)
    print("\n>>> 半双工输入 Token Schema（结构化 items）：")
    print(json.dumps([str(x) for x in input_schema_items], ensure_ascii=False, indent=2))

    # ---------- 4. Hook _decode 以捕获主干 LLM 输出 token ----------
    _orig_decode = model._decode
    captured_outputs = {}

    def _hooked_decode(inputs_embeds, tokenizer, attention_mask, **kwargs):
        outputs = _orig_decode(inputs_embeds, tokenizer, attention_mask, **kwargs)
        captured_outputs["outputs"] = outputs
        return outputs

    model._decode = _hooked_decode

    # ---------- 5. Hook TTS.generate 以捕获 VQ Token ----------
    _orig_tts_generate = model.tts.generate
    captured_tts_info = {}

    def _hooked_tts_generate(inputs_embeds, eos_token, force_no_stop=False,
                             min_new_token=50, max_new_token=2048,
                             show_tqdm=True, streaming=False,
                             text_lengths=None,
                             sampling_params=TTSSamplingParams()):
        print_section("Hook: TTS.generate (VQ Token 生成)")
        print_tensor_info("TTS inputs_embeds", inputs_embeds)
        print(f"    sampling_params: temp={sampling_params.temperature}, "
              f"top_p={sampling_params.top_p}, top_k={sampling_params.top_k}, "
              f"rep_penalty={sampling_params.repetition_penalty}")

        result = _orig_tts_generate(
            inputs_embeds=inputs_embeds, eos_token=eos_token,
            force_no_stop=force_no_stop, min_new_token=min_new_token,
            max_new_token=max_new_token, show_tqdm=show_tqdm,
            streaming=streaming, text_lengths=text_lengths,
            sampling_params=sampling_params,
        )

        print_section("Hook: TTS.generate 出口")
        if result.new_ids is not None:
            print_tensor_info("VQ new_ids", result.new_ids)
            if result.new_ids.dim() == 3:
                # [batch=1, seq_len, num_vq]
                vq_tokens_1st = result.new_ids[0, :, 0].cpu().tolist()
                print(f">>> VQ Token 第1码本序列 (len={len(vq_tokens_1st)})")
                print(f"    前50: {vq_tokens_1st[:50]}")
                print(f"    后50: {vq_tokens_1st[-50:]}")
                captured_tts_info["vq_tokens_1st"] = vq_tokens_1st
                captured_tts_info["vq_shape"] = list(result.new_ids.shape)
                captured_tts_info["finished"] = result.finished
            else:
                print(f">>> VQ new_ids shape 非3维，实际: {result.new_ids.shape}")
        else:
            print(">>> VQ new_ids 为 None")
        return result

    model.tts.generate = _hooked_tts_generate

    # ---------- 6. 调用 model.chat() 进行实际推理（复用原始路径） ----------
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

    # ---------- 7. 从捕获的输出中提取并打印输出 token ----------
    outputs = captured_outputs.get("outputs")
    if outputs is not None and hasattr(outputs, "sequences"):
        generated_ids = outputs.sequences[0]
        print_tokens("半双工主干 LLM 输出 Token IDs", tokenizer, generated_ids)
        output_schema_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        print("\n>>> 半双工主干 LLM 输出 Token Schema（文本占位符形式，完整）：")
        print(output_schema_text)

        # 拼接完整序列（input + output）
        full_seq = torch.cat([input_ids[0].cpu(), generated_ids.cpu()])
        full_seq_list = full_seq.tolist()

        # 标记 <|tts_bos|> / <|tts_eos|> 位置
        tts_bos_id = tokenizer.convert_tokens_to_ids("|<|tts_bos|>")
        tts_eos_id = tokenizer.convert_tokens_to_ids("|<|tts_eos|>")
        tts_bos_positions = [i for i, x in enumerate(full_seq_list) if x == tts_bos_id]
        tts_eos_positions = [i for i, x in enumerate(full_seq_list) if x == tts_eos_id]

        print(f"\n>>> 语音相关特殊 Token 位置分析：")
        print(f"    <|tts_bos|> (id={tts_bos_id}) 出现在位置: {tts_bos_positions}")
        print(f"    <|tts_eos|> (id={tts_eos_id}) 出现在位置: {tts_eos_positions}")

        # 提取 tts_bos 到 tts_eos 之间的文本（语音文本条件）
        if tts_bos_positions and tts_eos_positions:
            last_bos = tts_bos_positions[-1]
            last_eos = tts_eos_positions[-1]
            if last_bos < last_eos:
                tts_text_tokens = full_seq_list[last_bos:last_eos]
                print(f"\n>>> TTS 文本条件 Token (位置 {last_bos}~{last_eos})：")
                print_tokens("TTS 文本条件", tokenizer, tts_text_tokens, skip_special=False)

        # 打印 VQ Token 分析
        if captured_tts_info:
            print(f"\n>>> VQ Token 分析：")
            print(f"    VQ 形状: {captured_tts_info.get('vq_shape')}")
            print(f"    是否完成: {captured_tts_info.get('finished')}")
            vq_1st = captured_tts_info.get("vq_tokens_1st", [])
            print(f"    第1码本序列长度: {len(vq_1st)}")
            if len(vq_1st) > 0:
                print(f"    前30: {vq_1st[:30]}")
                print(f"    后30: {vq_1st[-30:]}")
        else:
            print("\n>>> [警告] 未能捕获 VQ Token 信息（可能未触发 TTS）")

        # ---------- 8. 保存中间结果 ----------
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
        print("[警告] 未能捕获 LLM 输出，可能 model.chat() 未触发 _decode")
        output_schema_text = ""
        generated_ids = torch.tensor([], dtype=torch.long)

    # ---------- 9. 解析 model.chat() 结果 ----------
    if isinstance(result, tuple):
        text, waveform = result
    else:
        text = result
        waveform = None

    return text, waveform


# ==================== 全双工模式（修复 TypeError） ====================

def run_full_duplex(model, tokenizer, audio_input, args):
    print_section("【全双工模式】处理流程")

    model.set_mode(ProcessorMode.DUPLEX)

    # ---------- 1. Prepare ----------
    ref_audio = audio_input
    prompt_wav_path = args.ref_audio or args.input_audio

    prefix = "|<|im_start|>system\nYou are a helpful assistant.\n<|<|audio_start|>"
    suffix = "|<|audio_end|>哦"

    full_prompt = model.duplex_prepare(
        prefix_system_prompt=prefix,
        suffix_system_prompt=suffix,
        ref_audio=ref_audio,
        prompt_wav_path=prompt_wav_path,
    )
    print(f">>> 全双工 System Prompt:\n{full_prompt}")

    # ---------- 2. 分块模拟实时流 ----------
    chunk_size = int(1.0 * 16000)  # 1s @ 16kHz
    total_len = len(audio_input)
    num_chunks = min(3, (total_len + chunk_size - 1) // chunk_size)

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_len)
        chunk = audio_input[start:end]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode="constant")

        print(f"\n>>> 全双工 Chunk {i + 1}/{num_chunks} (samples={len(chunk)})")
        prefill_res = model.duplex_prefill(audio_waveform=chunk, frame_list=None, max_slice_nums=1)
        print(f"    Prefill cost: {prefill_res.get('cost_all', 0):.3f}s")

        # 修复：显式传入所有参数，避免 None 覆盖 DuplexCapability 默认值导致 TypeError
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
        print(f"    Generate: is_listen={gen_res['is_listen']}, "
              f"text='{gen_res['text']}', end_of_turn={gen_res['end_of_turn']}")

        model.duplex_finalize()

    # ---------- 3. 获取完整 Schema（含占位符） ----------
    session_schema = model.duplex.get_session_schema(include_embeddings=True)
    unit_schemas = model.duplex.get_unit_schemas(include_embeddings=True)

    print("\n>>> 全双工会话 Schema（文本占位符形式，完整）：")
    print(session_schema)

    print(f"\n>>> 全双工 Unit 数量: {len(unit_schemas)}")
    for idx, us in enumerate(unit_schemas):
        print(f"\n--- Unit {idx} ---")
        print(us)

    # ---------- 4. 保存中间结果 ----------
    intermediate = {
        "mode": "full_duplex",
        "system_prompt": full_prompt,
        "session_schema": session_schema,
        "unit_schemas": unit_schemas,
        "prefill_schema_tokens": [
            [str(x) for x in unit] for unit in model.duplex.prefill_schema_tokens
        ],
        "total_ids": model.duplex.total_ids,
        "total_ids_text": tokenizer.decode(model.duplex.total_ids, skip_special_tokens=False) if model.duplex.total_ids else "",
    }
    save_intermediate(intermediate, "full_duplex_intermediate.json")

    return session_schema


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
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    args.generate_audio = True

    # 1. 修补并加载模型
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

    # 2. Unified 初始化
    print("[初始化] Unified 初始化 (chat_vocoder=token2wav) ...")
    model.init_unified(pt_path=None, chat_vocoder="token2wav", preload_both_tts=False)
    print("[初始化] 完成")

    tokenizer = model.processor.tokenizer

    # 3. 准备输入数据
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
    fd_schema = run_full_duplex(model, tokenizer, audio_input, args)

    # 6. 对比总结
    print_section("【对比总结】半双工 vs 全双工")
    print("【输入侧】")
    print("  半双工 (Half-Duplex / Chat):")
    print("    - 使用标准 Chat Template 结构：")
    print("      <|im_start|>system ... 哦 <|im_start|>user ... 哦 <|im_start|>assistant ...")
    print("    - Audio 区域被替换为连续占位符，如 [audio_embed_50]")
    print("    - Image 区域被替换为连续占位符，如 [img_embed_64]")
    print("    - 无 <unit> 包裹，无帧级切分概念")
    print()
    print("  全双工 (Full-Duplex / Duplex):")
    print("    - 每帧数据包裹在 <unit> ... </unit> 中：")
    print("      <unit> <image>[img_embed_64]</image> [audio_embed_50] <|listen|> </unit>")
    print("    - 若含 HD 图：含 <slice>[img_embed_64]</slice> 子结构")
    print("    - 参考音频嵌入在 system prompt 中作为独立 audio embed 块")
    print()
    print("【输出侧（主干 LLM）】")
    print("  半双工:")
    print("    - 标准自回归文本 token + 特殊标记：")
    print("      例如：你好呀！<|<|tts_bos|> ... <|tts_eos|> 哦")
    print("    - 无决策 token，无帧级终止符")
    print("    - TTS 侧生成 VQ Token（如 [626, 4218, 3000, ...]），由 hidden states 驱动")
    print()
    print("  全双工:")
    print("    - 每帧先生成决策 token：")
    print("      <|listen|>  -> 模型选择倾听（无文本输出）")
    print("      <|speak|>   -> 模型选择说话，后续跟随文本 token")
    print("    - 帧级终止符：")
    print("      <|chunk_eos|>  -> 当前 chunk 结束（未说完）")
    print("      <|turn_eos|>   -> 当前 turn 结束（说完）")
    print("    - 最后统一闭合：</unit>")
    print()
    print("【中间结果文件】")
    print("  - half_duplex_intermediate.json  （含 input_ids / audio_bounds / VQ token 样本）")
    print("  - full_duplex_intermediate.json  （含 prefill_schema_tokens / total_ids）")


if __name__ == "__main__":
    main()
