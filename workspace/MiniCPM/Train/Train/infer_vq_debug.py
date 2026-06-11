#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MiniCPM-O-4.5 推理调试脚本 (infer_debug.py) —— v9 完整版
放置于 Train 目录，与 MiniCPMO45 同级。

新增功能：
  1. 集成 VQCodec（s3tokenizer 编码 + Token2Wav 解码）
  2. 对输入音频编码得到 GT VQ tokens，与模型生成的 Pred VQ tokens 对比
  3. 用 GT tokens 额外解码一份音频 output_gt.wav，用于定位问题
  4. 支持 --replace-with-gt 将最终返回音频替换为 GT 解码（验证解码链路）

修复记录：
  - v8: Token2wav.stream() hook 参数名不匹配（*args, **kwargs 透传）
  - v8: processor 缺失 pixel_values / tgt_sizes 导致的 KeyError
  - v9: 集成 VQCodec 做 Pred/GT VQ tokens 对比与替换
"""

import argparse
import os
import sys
import json
import tempfile
from copy import deepcopy
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
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
    print(f"    Decode: {repr(text[:300])}{' ...' if len(text) > 300 else ''}")


def print_tensor_info(title: str, tensor: torch.Tensor):
    print(f"    {title}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}")


# ==================== Monkey-patch Hook ====================

def install_hooks(model, tokenizer, gt_vq_tokens=None, gt_audio_path=None, vq_codec=None, replace_with_gt=False):
    """
    在 model.chat() 调用的关键内部函数上安装 hook。
    新增：捕获 Pred VQ tokens，并与 GT VQ tokens 对比。
    """
    device = model.device

    # ---- Hook 1: _generate_speech_non_streaming ----
    _orig_generate_speech = model._generate_speech_non_streaming

    def _hooked_generate_speech(outputs, tts_bound, tts_proj_layer, audio_prompt,
                                output_tts_inputs_embeds_path=None,
                                tts_sampling_params=TTSSamplingParams()):
        print_section("Hook: _generate_speech_non_streaming 入口")
        tts_bos_idx, tts_eos_idx = tts_bound
        print(f">>> tts_bound = ({tts_bos_idx}, {tts_eos_idx})")

        if "full_sequences" in outputs:
            full_seq = outputs["full_sequences"][0]
            print_tokens("full_sequences (来自 outputs)", tokenizer, full_seq, skip_special=False)
            if tts_bos_idx >= 0:
                tts_slice = full_seq[tts_bos_idx:tts_eos_idx]
                print_tokens("tts_bound 内文本 token", tokenizer, tts_slice, skip_special=False)

        last_hidden_states = [hs[tts_proj_layer] for hs in outputs.hidden_states]
        last_hidden_states = torch.vstack([i[0] for i in last_hidden_states])
        full_seq_len = len(outputs["full_sequences"][0]) if "full_sequences" in outputs else "N/A"
        print_tensor_info("last_hidden_states (堆叠后)", last_hidden_states)
        print(f"    full_sequences len = {full_seq_len}")

        # 调用原函数（Pred 音频）
        result = _orig_generate_speech(
            outputs=outputs, tts_bound=tts_bound, tts_proj_layer=tts_proj_layer,
            audio_prompt=audio_prompt, output_tts_inputs_embeds_path=output_tts_inputs_embeds_path,
            tts_sampling_params=tts_sampling_params,
        )

        print_section("Hook: _generate_speech_non_streaming 出口")
        pred_waveform = result
        if isinstance(pred_waveform, np.ndarray):
            print(f">>> Pred 返回波形: {len(pred_waveform)} samples, {len(pred_waveform)/24000:.2f}s @ 24kHz")
        elif isinstance(pred_waveform, torch.Tensor):
            print(f">>> Pred 返回波形 tensor: {tuple(pred_waveform.shape)}")
        else:
            print(f">>> Pred 返回类型: {type(pred_waveform)}")

        # ========== VQ Pred vs GT 对比 & 替换 ==========
        if gt_vq_tokens is not None and vq_codec is not None:
            pred_tokens = getattr(model, '_last_pred_vq_tokens', None)

            print_section("VQ Tokens 对比: Pred vs GT")
            gt_list = list(gt_vq_tokens)

            if pred_tokens is not None:
                if isinstance(pred_tokens, torch.Tensor):
                    pred_tokens = pred_tokens.detach().cpu()
                    if pred_tokens.dim() == 3:
                        pred_tokens = pred_tokens[0, :, 0]  # (seq, num_vq) -> 取第一码本
                    pred_list = pred_tokens.reshape(-1).tolist()
                else:
                    pred_list = list(pred_tokens)

                print(f"    Pred tokens: {len(pred_list)} | GT tokens: {len(gt_list)}")
                min_len = min(len(pred_list), len(gt_list))
                if min_len > 0:
                    matches = sum(1 for a, b in zip(pred_list[:min_len], gt_list[:min_len]) if a == b)
                    print(f"    前 {min_len} 个重合: {matches}/{min_len} ({100*matches/min_len:.1f}%)")
                    print(f"    长度差异: Pred - GT = {len(pred_list) - len(gt_list)}")
                    diffs = [(i, p, g) for i, (p, g) in enumerate(zip(pred_list[:min_len], gt_list[:min_len])) if p != g]
                    if diffs:
                        print(f"    差异示例 (前10处 idx/pred/gt): {diffs[:10]}")
                    else:
                        print(f"    前 {min_len} 个 token 完全一致")
                else:
                    print("    无法对比（某一方为空）")

                # 分布统计
                pred_unique = len(set(pred_list))
                gt_unique = len(set(gt_list))
                print(f"    Pred unique tokens: {pred_unique} | GT unique tokens: {gt_unique}")
            else:
                print("    [警告] 未捕获到 Pred tokens（TTS.generate hook 未触发）")

            # 用 GT tokens 解码一份音频，用于对比
            try:
                print_section("GT VQ Tokens 解码")
                gt_waveform = vq_codec.decode(
                    gt_vq_tokens,
                    prompt_wav_path=gt_audio_path,
                    add_silence_prefix=True,
                )
                print(f"    GT 解码音频: {len(gt_waveform)} samples ({len(gt_waveform)/24000:.2f}s @ 24kHz)")

                # 保存 GT 音频（固定文件名或基于 output_audio）
                gt_out_path = os.path.join(os.path.dirname(args.output_audio), "output_gt.wav") if hasattr(args, 'output_audio') else "output_gt.wav"
                sf.write(gt_out_path, gt_waveform, 24000)
                print(f"    [保存] GT 音频 -> {gt_out_path}")

                if replace_with_gt:
                    print("    [替换] 返回 GT 音频替代 Pred 音频（--replace-with-gt）")
                    result = gt_waveform
            except Exception as e:
                print(f"    [错误] GT 解码失败: {e}")
                import traceback
                traceback.print_exc()

        return result

    model._generate_speech_non_streaming = _hooked_generate_speech

    # ---- Hook 2: MiniCPMTTS.generate ----
    if hasattr(model, 'tts') and model.tts is not None:
        _orig_tts_generate = model.tts.generate

        def _hooked_tts_generate(inputs_embeds, eos_token, force_no_stop=False,
                                 min_new_token=50, max_new_token=2048,
                                 show_tqdm=True, streaming=False,
                                 text_lengths=None,
                                 sampling_params=TTSSamplingParams()):
            print_section("Hook: MiniCPMTTS.generate 入口")
            print_tensor_info("inputs_embeds (TTS 输入)", inputs_embeds)
            print(f"    sampling_params: temp={sampling_params.temperature}, "
                  f"top_p={sampling_params.top_p}, top_k={sampling_params.top_k}")

            result = _orig_tts_generate(
                inputs_embeds=inputs_embeds, eos_token=eos_token,
                force_no_stop=force_no_stop, min_new_token=min_new_token,
                max_new_token=max_new_token, show_tqdm=show_tqdm,
                streaming=streaming, text_lengths=text_lengths,
                sampling_params=sampling_params,
            )

            print_section("Hook: MiniCPMTTS.generate 出口")
            new_ids = result.new_ids
            # 保存 pred tokens 供上层对比
            model._last_pred_vq_tokens = new_ids.detach().cpu() if new_ids is not None else None

            print_tensor_info("new_ids (VQ Tokens)", new_ids)
            if new_ids is not None and new_ids.numel() > 0:
                if new_ids.dim() == 3:
                    tokens_1st = new_ids[0, :, 0].cpu().tolist()
                else:
                    tokens_1st = new_ids.reshape(-1).tolist()
                print(f">>> VQ Token 序列 (第1码本, len={len(tokens_1st)})")
                print(f"    前50: {tokens_1st[:50]}")
                print(f"    后50: {tokens_1st[-50:]}")
            return result

        model.tts.generate = _hooked_tts_generate

    # ---- Hook 3: Token2wav.stream ----
    if (hasattr(model, 'tts') and model.tts is not None and
            hasattr(model.tts, 'audio_tokenizer') and model.tts.audio_tokenizer is not None):
        tokenizer_obj = model.tts.audio_tokenizer
        if hasattr(tokenizer_obj, 'stream'):
            _orig_t2w_stream = tokenizer_obj.stream

            def _hooked_t2w_stream(*args, **kwargs):
                token_ids = args[0] if args else (kwargs.get('token_ids') or kwargs.get('tokens'))
                last_chunk = kwargs.get('last_chunk', False)
                return_waveform = kwargs.get('return_waveform', True)

                _tid = token_ids if hasattr(token_ids, '__len__') else []
                print(f"\n    [Token2Wav.stream] len={len(_tid)}, last_chunk={last_chunk}, "
                      f"return_waveform={return_waveform}")
                if len(_tid) > 0:
                    print(f"        token_ids[:20]: {list(_tid)[:20]}")
                    print(f"        token_ids[-20:]: {list(_tid)[-20:]}")

                result = _orig_t2w_stream(*args, **kwargs)
                if result is not None and return_waveform:
                    wav = result.squeeze() if hasattr(result, 'squeeze') else result
                    print(f"        -> 输出波形: {len(wav)} samples")
                return result

            tokenizer_obj.stream = _hooked_t2w_stream

    print("[Hook] 已安装 _generate_speech_non_streaming / TTS.generate / Token2Wav.stream 打印钩子")


# ==================== 独立分析：输入 Token 结构 ====================

def analyze_input_structure(model, msgs, use_tts_template=True):
    print_section("独立分析：输入 Token 结构（Processor + Chat Template）")

    processor = model.processor
    tokenizer = processor.tokenizer

    copy_msgs = deepcopy(msgs)
    images = []
    audios = []
    audio_parts = []

    for msg in copy_msgs:
        content = msg["content"]
        if isinstance(content, str):
            content = [content]
        cur_msgs = []
        for c in content:
            if isinstance(c, np.ndarray):
                audios.append(c)
                audio_parts.append(0)
                cur_msgs.append("<audio>./</audio>")
                use_tts_template = True
            elif isinstance(c, str):
                cur_msgs.append(c)
        msg["content"] = "\n".join(cur_msgs)

    prompt = tokenizer.apply_chat_template(
        copy_msgs,
        tokenize=False,
        add_generation_prompt=True,
        use_tts_template=use_tts_template,
    )

    print("\n>>> Prompt 字符串:")
    print(prompt)

    inputs = processor(
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
            print(f"    [兜底] inputs 缺失 '{key}'，已设为 None")

    print(">>> processor 输出 keys:", list(inputs.keys()))

    input_ids = inputs["input_ids"][0]
    print_tokens("input_ids (完整 Prompt)", tokenizer, input_ids, skip_special=False)

    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("")
    tts_bos = tokenizer.convert_tokens_to_ids("<|tts_bos|>")
    tts_eos = tokenizer.convert_tokens_to_ids("<|tts_eos|>")

    ids = input_ids.cpu().tolist()
    segments = []
    current_start = 0

    for i, tid in enumerate(ids):
        if tid == im_start:
            current_start = i
        elif tid == im_end and current_start is not None:
            seg_ids = ids[current_start:i+1]
            role = tokenizer.decode([ids[current_start+1]] if current_start+1 < len(ids) else [], skip_special_tokens=True)
            segments.append({
                "role": role,
                "start": current_start,
                "end": i,
                "len": i - current_start + 1,
                "text_preview": tokenizer.decode(seg_ids, skip_special_tokens=True)[:100]
            })
            current_start = None

    print("\n>>> 分段结构:")
    for seg in segments:
        print(f"    [{seg['role']}] idx={seg['start']}~{seg['end']} (len={seg['len']})")
        print(f"        preview: {repr(seg['text_preview'])}")

    if inputs.get("audio_bounds") is not None:
        print(f"\n>>> audio_bounds: {inputs['audio_bounds'][0].tolist()}")
        for bound in inputs["audio_bounds"][0].tolist():
            start, end = bound
            print(f"    音频占位符位置: [{start}, {end}) -> token IDs: {ids[start:end]}")
            print(f"        decode: {repr(tokenizer.decode(ids[start:end], skip_special_tokens=False))}")

    pixel_values = inputs.get("pixel_values")
    if pixel_values is None:
        pixel_values = [[]]
    tgt_sizes = inputs.get("tgt_sizes")
    if tgt_sizes is None:
        tgt_sizes = [[]]

    model_inputs = {
        "input_ids": inputs["input_ids"],
        "audio_features": inputs.get("audio_features"),
        "audio_feature_lens": inputs.get("audio_feature_lens"),
        "image_bound": inputs.get("image_bound"),
        "audio_bounds": inputs.get("audio_bounds"),
        "spk_bounds": inputs.get("spk_bounds"),
        "pixel_values": pixel_values,
        "tgt_sizes": tgt_sizes,
    }

    vllm_embedding, _ = model.get_vllm_embedding(model_inputs)
    inputs_embeds = model.get_omni_embedding(
        model_inputs, input_embeddings=vllm_embedding,
        chunk_length=model.config.audio_chunk_length,
    )
    print(f"\n>>> inputs_embeds shape: {tuple(inputs_embeds.shape)}")
    print(f"    说明: [batch=1, seq_len={inputs_embeds.shape[1]}, hidden_dim={inputs_embeds.shape[2]}]")

    print("\n>>> 建议 Loss Mask 规则:")
    print("    - System/User 段 (prompt): mask = 0")
    print("    - Assistant 生成段: mask = 1")
    print("    - 特殊结构 token (<|tts_bos|>, <|tts_eos|>): mask = 1")
    print("    - Audio VQ Token (训练时由 TTS 生成): mask = 1")

    return {
        "prompt": prompt,
        "input_ids": ids,
        "segments": segments,
        "inputs_embeds_shape": tuple(inputs_embeds.shape),
    }


# ==================== CLI 入口 ====================

def parse_args():
    parser = argparse.ArgumentParser(description="MiniCPM-O-4.5 Unified Debug Inference (Hook版)")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--input-audio", type=str, required=True)
    parser.add_argument("--input-text", type=str, default="")
    parser.add_argument("--output-text", type=str, default="output.txt")
    parser.add_argument("--output-audio", type=str, default="output.wav")
    parser.add_argument("--ref-audio", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--tts-temperature", type=float, default=0.8)
    parser.add_argument("--tts-top-p", type=float, default=0.85)
    parser.add_argument("--tts-top-k", type=int, default=25)
    parser.add_argument("--replace-with-gt", action="store_true",
                        help="将最终输出音频替换为 GT VQ tokens 解码音频（用于验证解码链路）")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    # 1. 修补 config
    print(f"[初始化] 读取并修补 config: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    if hasattr(config, 'tts_config') and config.tts_config is not None:
        for attr, default in [('top_p', 0.85), ('top_k', 25), ('repetition_penalty', 1.05)]:
            if not hasattr(config.tts_config, attr):
                setattr(config.tts_config, attr, default)
                print(f"    [修复] config.tts_config.{attr} = {default}")

    # 2. 加载模型
    print(f"[初始化] 加载模型 (dtype={args.dtype}, device={args.device}) ...")
    model = MiniCPMO.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device,
    )
    model.eval()

    # 3. 初始化 Unified
    print("[初始化] Unified 初始化 (chat_vocoder=token2wav) ...")
    model.init_unified(pt_path=None, chat_vocoder="token2wav", preload_both_tts=False)
    model.set_mode(ProcessorMode.CHAT)
    print("[初始化] 完成，进入 CHAT 模式")

    # 4. 准备输入音频
    audio_input, sr = librosa.load(args.input_audio, sr=16000, mono=True)
    print(f"[输入] 加载音频: {args.input_audio}, 时长={len(audio_input)/16000:.2f}s")

    tts_ref_audio = None
    if args.ref_audio:
        tts_ref_audio, _ = librosa.load(args.ref_audio, sr=16000, mono=True)
    else:
        tts_ref_audio = audio_input

    # 5. 用 VQCodec 编码输入音频，得到 GT VQ tokens（用于对比）
    gt_vq_tokens = None
    vq_codec = None
    gt_audio_path = None

    try:
        from vq_codec import VQCodec
        print("[VQCodec] 初始化...")
        vq_codec = VQCodec(model_dir=args.model_path, device=args.device)

        # 编码输入音频（作为 GT）
        gt_vq_tokens = vq_codec.encode(audio_input)
        print(f"[VQCodec] GT VQ tokens: {len(gt_vq_tokens)} tokens, "
              f"range [{int(gt_vq_tokens.min())}, {int(gt_vq_tokens.max())}], "
              f"unique={len(set(gt_vq_tokens))}")

        # 保存临时参考音频供 decode 使用
        gt_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(gt_tmp.name, audio_input, 16000)
        gt_audio_path = gt_tmp.name
        print(f"[VQCodec] 临时参考音频: {gt_audio_path}")
    except Exception as e:
        print(f"[警告] VQCodec 初始化/编码失败，跳过 GT 对比: {e}")

    # 6. 安装 Hook（传入 GT tokens 供对比）
    install_hooks(
        model,
        model.processor.tokenizer,
        gt_vq_tokens=gt_vq_tokens,
        gt_audio_path=gt_audio_path,
        vq_codec=vq_codec,
        replace_with_gt=args.replace_with_gt,
    )

    # 7. 构建消息
    user_content = [audio_input]
    if args.input_text:
        user_content.append(args.input_text)

    msgs = [
        {"role": "system", "content": [args.system_prompt]},
        {"role": "user", "content": user_content},
    ]

    # 8. 独立分析输入结构
    analyze_input_structure(model, msgs, use_tts_template=True)

    # 9. 运行推理
    print_section("开始 model.chat() 推理（Hook 将自动打印内部数据）")

    tts_sampling_params = TTSSamplingParams(
        temperature=args.tts_temperature,
        top_p=args.tts_top_p,
        top_k=args.tts_top_k,
    )

    result = model.chat(
        msgs=msgs,
        generate_audio=True,
        use_tts_template=True,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        tts_ref_audio=tts_ref_audio,
        tts_sampling_params=tts_sampling_params,
    )

    # 10. 解析结果
    if isinstance(result, tuple):
        text, waveform = result
    else:
        text = result
        waveform = None

    with open(args.output_text, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n[保存] 文本 -> {args.output_text}")
    print(f"文本内容:\n{text}")

    if waveform is not None:
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        sf.write(args.output_audio, waveform, samplerate=24000)
        print(f"[保存] 音频 -> {args.output_audio} ({len(waveform)/24000:.2f}s @ 24kHz)")
    else:
        print("[保存] 无音频输出")

    # 11. 清理临时文件
    if gt_audio_path and os.path.exists(gt_audio_path):
        try:
            os.unlink(gt_audio_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
