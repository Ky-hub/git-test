#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MiniCPM-O-4.5 推理调试脚本 (infer_debug.py) —— v13
核心改动：
  - model.chat() 返回的音频不再走模型内置 Token2Wav，而是提取 Pred VQ tokens
    交给用户 VQCodec 解码，生成 output.wav
  - 同时保留模型原始解码音频为 output_orig.wav（用于对比）
  - GT 编码解码生成 output_gt.wav（闭环验证）
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


# ==================== 日志工具（强制落盘） ====================

_LOG_FILE = None

def log_init(path="infer_debug.log"):
    global _LOG_FILE
    _LOG_FILE = open(path, "w", encoding="utf-8")
    print(f"[日志] 同时写入文件: {path}", flush=True)

def log_flush(msg: str):
    print(msg, flush=True)
    if _LOG_FILE:
        _LOG_FILE.write(msg + "\n")
        _LOG_FILE.flush()

def log_section(title: str):
    s = "\n" + "=" * 80 + f"\n  {title}\n" + "=" * 80
    log_flush(s)

def log_tokens(title: str, tokenizer, ids, skip_special: bool = False, limit: int = 100):
    if isinstance(ids, torch.Tensor):
        ids = ids.cpu().tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = list(ids)
    text = tokenizer.decode(ids, skip_special_tokens=skip_special)
    log_flush(f"\n>>> {title}  (len={len(ids)})")
    log_flush(f"    IDs: {ids[:limit]}{' ...' if len(ids) > limit else ''}")
    log_flush(f"    Decode: {repr(text[:300])}{' ...' if len(text) > 300 else ''}")

def log_tensor(title: str, tensor):
    if tensor is None:
        log_flush(f"    {title}: None")
    elif isinstance(tensor, torch.Tensor):
        log_flush(f"    {title}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}")
    else:
        log_flush(f"    {title}: type={type(tensor)}, value={tensor}")


# ==================== Monkey-patch Hook ====================

def install_hooks(model, tokenizer, gt_vq_tokens=None, gt_audio_path=None,
                  vq_codec=None, replace_with_gt=False, output_audio_path="output.wav"):
    """
    在 model.chat() 调用的关键内部函数上安装 hook。
    核心：_generate_speech_non_streaming 出口用 VQCodec 解码 Pred tokens，替换模型原始音频。
    """

    # ---- 通用：保存 TTS 生成的 token ----
    def _save_pred_tokens(new_ids):
        if new_ids is None:
            model._last_pred_vq_tokens = None
            return
        t = new_ids.detach().cpu()
        if t.dim() == 3:
            t = t[0, :, 0]  # (seq, num_vq) -> 第一码本
        model._last_pred_vq_tokens = t.reshape(-1)
        log_flush(f"[TokenCapture] 捕获 Pred tokens: {model._last_pred_vq_tokens.shape}, "
                  f"range [{int(model._last_pred_vq_tokens.min())}, {int(model._last_pred_vq_tokens.max())}]")

    # Hook: tts.generate (非 interleaved)
    if hasattr(model, 'tts') and model.tts is not None:
        _orig_tts_generate = model.tts.generate
        def _hooked_tts_generate(*args, **kwargs):
            log_section("Hook: MiniCPMTTS.generate 入口")
            if "inputs_embeds" in kwargs:
                emb = kwargs["inputs_embeds"]
            elif args:
                emb = args[0]
            else:
                emb = None
            log_tensor("inputs_embeds", emb)
            result = _orig_tts_generate(*args, **kwargs)
            log_section("Hook: MiniCPMTTS.generate 出口")
            log_tensor("new_ids", result.new_ids)
            _save_pred_tokens(result.new_ids)
            return result
        model.tts.generate = _hooked_tts_generate

        # Hook: tts.interleaved_generate
        if hasattr(model.tts, 'interleaved_generate'):
            _orig_interleaved = model.tts.interleaved_generate
            def _hooked_interleaved(*args, **kwargs):
                log_section("Hook: MiniCPMTTS.interleaved_generate 入口")
                result = _orig_interleaved(*args, **kwargs)
                log_section("Hook: MiniCPMTTS.interleaved_generate 出口")
                log_tensor("new_ids", result.new_ids)
                _save_pred_tokens(result.new_ids)
                return result
            model.tts.interleaved_generate = _hooked_interleaved

        # Hook: tts.generate_chunk
        if hasattr(model.tts, 'generate_chunk'):
            _orig_chunk = model.tts.generate_chunk
            def _hooked_chunk(*args, **kwargs):
                log_section("Hook: MiniCPMTTS.generate_chunk 入口")
                gen_ids, past_kv = _orig_chunk(*args, **kwargs)
                log_section("Hook: MiniCPMTTS.generate_chunk 出口")
                if gen_ids is not None:
                    log_tensor("gen_ids", gen_ids)
                    _save_pred_tokens(gen_ids)
                return gen_ids, past_kv
            model.tts.generate_chunk = _hooked_chunk

    # ---- Hook: _generate_speech_non_streaming ----
    _orig_generate_speech = model._generate_speech_non_streaming

    def _hooked_generate_speech(outputs, tts_bound, tts_proj_layer, audio_prompt,
                                output_tts_inputs_embeds_path=None,
                                tts_sampling_params=TTSSamplingParams()):
        log_section("Hook: _generate_speech_non_streaming 入口")
        tts_bos_idx, tts_eos_idx = tts_bound
        log_flush(f">>> tts_bound = ({tts_bos_idx}, {tts_eos_idx})")

        if "full_sequences" in outputs:
            full_seq = outputs["full_sequences"][0]
            log_tokens("full_sequences", tokenizer, full_seq, skip_special=False)
            if tts_bos_idx >= 0 and tts_eos_idx is not None:
                tts_slice = full_seq[tts_bos_idx:tts_eos_idx]
                log_tokens("tts_bound 内文本 token", tokenizer, tts_slice, skip_special=False)

        last_hidden_states = [hs[tts_proj_layer] for hs in outputs.hidden_states]
        last_hidden_states = torch.vstack([i[0] for i in last_hidden_states])
        full_seq_len = len(outputs["full_sequences"][0]) if "full_sequences" in outputs else "N/A"
        log_tensor("last_hidden_states (堆叠后)", last_hidden_states)
        log_flush(f"    full_sequences len = {full_seq_len}")

        # 1. 先调用原函数，得到模型原始解码音频（用于对比）
        result_orig = _orig_generate_speech(
            outputs=outputs, tts_bound=tts_bound, tts_proj_layer=tts_proj_layer,
            audio_prompt=audio_prompt, output_tts_inputs_embeds_path=output_tts_inputs_embeds_path,
            tts_sampling_params=tts_sampling_params,
        )

        log_section("Hook: _generate_speech_non_streaming 出口")
        if isinstance(result_orig, np.ndarray):
            log_flush(f">>> 模型原始波形: {len(result_orig)} samples, {len(result_orig)/24000:.2f}s @ 24kHz")
        elif isinstance(result_orig, torch.Tensor):
            log_flush(f">>> 模型原始波形 tensor: {tuple(result_orig.shape)}")
        else:
            log_flush(f">>> 模型原始返回类型: {type(result_orig)}")

        # 2. 保存模型原始解码音频（对比用）
        orig_path = os.path.join(os.path.dirname(output_audio_path), "output_orig.wav") if output_audio_path else "output_orig.wav"
        if isinstance(result_orig, np.ndarray):
            sf.write(orig_path, result_orig, 24000)
            log_flush(f"    [保存] 模型原始音频 -> {orig_path}")

        # 3. 提取 Pred tokens
        pred_tokens = getattr(model, '_last_pred_vq_tokens', None)
        log_flush(f"\n[对比检查] pred_tokens 类型: {type(pred_tokens)}, gt_vq_tokens 类型: {type(gt_vq_tokens)}")

        # 4. 用 VQCodec 解码 Pred tokens -> 替换 result
        result_vqcodec = None
        if vq_codec is not None and pred_tokens is not None and pred_tokens.numel() > 0:
            try:
                log_section("VQCodec 解码 Pred tokens")
                pred_list = pred_tokens.cpu().numpy()  # VQCodec 接受 np.ndarray / list / tensor
                log_flush(f"    Pred tokens 长度: {len(pred_list)}, range [{int(pred_list.min())}, {int(pred_list.max())}]")

                # 兼容 VQCodec 是否支持 add_silence_prefix
                try:
                    result_vqcodec = vq_codec.decode(
                        pred_list,
                        prompt_wav_path=gt_audio_path,
                        add_silence_prefix=True,
                    )
                except TypeError as te:
                    if "add_silence_prefix" in str(te):
                        log_flush("    [兼容] VQCodec.decode 不支持 add_silence_prefix，重新调用...")
                        result_vqcodec = vq_codec.decode(
                            pred_list,
                            prompt_wav_path=gt_audio_path,
                        )
                    else:
                        raise

                log_flush(f"    VQCodec 解码音频: {len(result_vqcodec)} samples ({len(result_vqcodec)/24000:.2f}s @ 24kHz)")

                # 保存 VQCodec 解码的 Pred 音频
                vq_pred_path = os.path.join(os.path.dirname(output_audio_path), "output_vq_pred.wav") if output_audio_path else "output_vq_pred.wav"
                sf.write(vq_pred_path, result_vqcodec, 24000)
                log_flush(f"    [保存] VQCodec Pred 音频 -> {vq_pred_path}")

            except Exception as e:
                log_flush(f"    [错误] VQCodec 解码 Pred tokens 失败: {e}")
                import traceback
                traceback.print_exc()

        # 5. 用 VQCodec 解码 GT tokens（闭环验证）
        if vq_codec is not None and gt_vq_tokens is not None:
            try:
                log_section("VQCodec 解码 GT tokens")
                try:
                    gt_waveform = vq_codec.decode(
                        gt_vq_tokens,
                        prompt_wav_path=gt_audio_path,
                        add_silence_prefix=True,
                    )
                except TypeError as te:
                    if "add_silence_prefix" in str(te):
                        gt_waveform = vq_codec.decode(
                            gt_vq_tokens,
                            prompt_wav_path=gt_audio_path,
                        )
                    else:
                        raise

                log_flush(f"    GT 解码音频: {len(gt_waveform)} samples ({len(gt_waveform)/24000:.2f}s @ 24kHz)")

                gt_out_path = os.path.join(os.path.dirname(output_audio_path), "output_gt.wav") if output_audio_path else "output_gt.wav"
                sf.write(gt_out_path, gt_waveform, 24000)
                log_flush(f"    [保存] GT 音频 -> {gt_out_path}")

            except Exception as e:
                log_flush(f"    [错误] GT 解码失败: {e}")
                import traceback
                traceback.print_exc()

        # 6. 决定返回哪个音频
        # 优先级：VQCodec Pred > 模型原始 > None
        if result_vqcodec is not None:
            log_flush("\n[返回] 使用 VQCodec 解码的 Pred 音频作为 model.chat() 返回值")
            return result_vqcodec
        else:
            log_flush("\n[返回] 使用模型原始解码音频（VQCodec 解码失败 fallback）")
            return result_orig

    model._generate_speech_non_streaming = _hooked_generate_speech

    # ---- Hook: Token2wav.stream（仅打印，不替换） ----
    if (hasattr(model, 'tts') and model.tts is not None and
            hasattr(model.tts, 'audio_tokenizer') and model.tts.audio_tokenizer is not None):
        tokenizer_obj = model.tts.audio_tokenizer
        if hasattr(tokenizer_obj, 'stream'):
            _orig_t2w_stream = tokenizer_obj.stream
            def _hooked_t2w_stream(*args, **kwargs):
                if args:
                    token_ids = args[0]
                elif 'token_ids' in kwargs:
                    token_ids = kwargs['token_ids']
                elif 'tokens' in kwargs:
                    token_ids = kwargs['tokens']
                else:
                    token_ids = None

                last_chunk = kwargs.get('last_chunk', False)
                _tid = token_ids if hasattr(token_ids, '__len__') else []
                log_flush(f"\n[Token2Wav.stream] len={len(_tid)}, last_chunk={last_chunk}")
                if len(_tid) > 0:
                    log_flush(f"    token_ids[:20]: {list(_tid)[:20]}")
                    log_flush(f"    token_ids[-20:]: {list(_tid)[-20:]}")
                result = _orig_t2w_stream(*args, **kwargs)
                if result is not None:
                    wav = result.squeeze() if hasattr(result, 'squeeze') else result
                    log_flush(f"    -> 输出波形: {len(wav)} samples")
                return result
            tokenizer_obj.stream = _hooked_t2w_stream

    log_flush("[Hook] 安装完成。输出音频优先级: VQCodec Pred > 模型原始")


# ==================== 独立分析：输入 Token 结构 ====================

def analyze_input_structure(model, msgs, use_tts_template=True):
    log_section("独立分析：输入 Token 结构（Processor + Chat Template）")

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

    log_flush("\n>>> Prompt 字符串:")
    log_flush(prompt)

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
            log_flush(f"    [兜底] inputs 缺失 '{key}'，已设为 None")

    log_flush(">>> processor 输出 keys: " + str(list(inputs.keys())))

    input_ids = inputs["input_ids"][0]
    log_tokens("input_ids (完整 Prompt)", tokenizer, input_ids, skip_special=False)

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

    log_flush("\n>>> 分段结构:")
    for seg in segments:
        log_flush(f"    [{seg['role']}] idx={seg['start']}~{seg['end']} (len={seg['len']})")
        log_flush(f"        preview: {repr(seg['text_preview'])}")

    if inputs.get("audio_bounds") is not None:
        log_flush(f"\n>>> audio_bounds: {inputs['audio_bounds'][0].tolist()}")
        for bound in inputs["audio_bounds"][0].tolist():
            start, end = bound
            log_flush(f"    音频占位符位置: [{start}, {end}) -> token IDs: {ids[start:end]}")
            log_flush(f"        decode: {repr(tokenizer.decode(ids[start:end], skip_special_tokens=False))}")

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
    log_flush(f"\n>>> inputs_embeds shape: {tuple(inputs_embeds.shape)}")
    log_flush(f"    说明: [batch=1, seq_len={inputs_embeds.shape[1]}, hidden_dim={inputs_embeds.shape[2]}]")

    log_flush("\n>>> 建议 Loss Mask 规则:")
    log_flush("    - System/User 段 (prompt): mask = 0")
    log_flush("    - Assistant 生成段: mask = 1")
    log_flush("    - 特殊结构 token (<|tts_bos|>, <|tts_eos|>): mask = 1")
    log_flush("    - Audio VQ Token (训练时由 TTS 生成): mask = 1")

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
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    log_init("infer_debug.log")

    log_flush(f"[初始化] 读取并修补 config: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    if hasattr(config, 'tts_config') and config.tts_config is not None:
        for attr, default in [('top_p', 0.85), ('top_k', 25), ('repetition_penalty', 1.05)]:
            if not hasattr(config.tts_config, attr):
                setattr(config.tts_config, attr, default)
                log_flush(f"    [修复] config.tts_config.{attr} = {default}")

    log_flush(f"[初始化] 加载模型 (dtype={args.dtype}, device={args.device}) ...")
    model = MiniCPMO.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device,
    )
    model.eval()

    log_flush("[初始化] Unified 初始化 (chat_vocoder=token2wav) ...")
    model.init_unified(pt_path=None, chat_vocoder="token2wav", preload_both_tts=False)
    model.set_mode(ProcessorMode.CHAT)
    log_flush("[初始化] 完成，进入 CHAT 模式")

    audio_input, sr = librosa.load(args.input_audio, sr=16000, mono=True)
    log_flush(f"[输入] 加载音频: {args.input_audio}, 时长={len(audio_input)/16000:.2f}s")

    tts_ref_audio = None
    if args.ref_audio:
        tts_ref_audio, _ = librosa.load(args.ref_audio, sr=16000, mono=True)
    else:
        tts_ref_audio = audio_input

    gt_vq_tokens = None
    vq_codec = None
    gt_audio_path = None

    try:
        from vq_codec import VQCodec
        log_flush("[VQCodec] 初始化...")
        vq_codec = VQCodec(model_dir=args.model_path, device=args.device)

        gt_vq_tokens = vq_codec.encode(audio_input)
        log_flush(f"[VQCodec] GT VQ tokens: {len(gt_vq_tokens)} tokens, "
                  f"range [{int(gt_vq_tokens.min())}, {int(gt_vq_tokens.max())}], "
                  f"unique={len(set(gt_vq_tokens))}")

        gt_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(gt_tmp.name, audio_input, 16000)
        gt_audio_path = gt_tmp.name
        log_flush(f"[VQCodec] 临时参考音频: {gt_audio_path}")
    except Exception as e:
        log_flush(f"[警告] VQCodec 初始化/编码失败: {e}")
        import traceback
        traceback.print_exc()

    install_hooks(
        model,
        model.processor.tokenizer,
        gt_vq_tokens=gt_vq_tokens,
        gt_audio_path=gt_audio_path,
        vq_codec=vq_codec,
        output_audio_path=args.output_audio,
    )

    user_content = [audio_input]
    if args.input_text:
        user_content.append(args.input_text)

    msgs = [
        {"role": "system", "content": [args.system_prompt]},
        {"role": "user", "content": user_content},
    ]

    analyze_input_structure(model, msgs, use_tts_template=True)

    log_section("开始 model.chat() 推理")

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

    if isinstance(result, tuple):
        text, waveform = result
    else:
        text = result
        waveform = None

    with open(args.output_text, "w", encoding="utf-8") as f:
        f.write(text)
    log_flush(f"\n[保存] 文本 -> {args.output_text}")
    log_flush(f"文本内容:\n{text}")

    if waveform is not None:
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        sf.write(args.output_audio, waveform, samplerate=24000)
        log_flush(f"[保存] 音频 -> {args.output_audio} ({len(waveform)/24000:.2f}s @ 24kHz)")
    else:
        log_flush("[保存] 无音频输出")

    if gt_audio_path and os.path.exists(gt_audio_path):
        try:
            os.unlink(gt_audio_path)
        except OSError:
            pass

    if _LOG_FILE:
        _LOG_FILE.close()


if __name__ == "__main__":
    main()
