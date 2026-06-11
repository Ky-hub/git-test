#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MiniCPM-O-4.5 (Unified) 推理脚本 infer.py
放置于 Train 目录下，与 MiniCPMO45 同级，不修改 MiniCPMO45 包内任何文件。

目录结构:
    parent/
      ├── MiniCPMO45/              # 模型包（不修改）
      │     ├── modeling_minicpmo_unified.py
      │     └── ...
      └── Train/
            └── infer.py           # 本脚本

功能: 给定 system prompt + 输入语音，输出回复文本与语音。
"""

import argparse
import os
import sys

# 将上级目录加入 sys.path，以便导入同级的 MiniCPMO45
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import librosa
import soundfile as sf

# 关键修正: 从 unified 版本导入（而非 modeling_minicpmo）
from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO, ProcessorMode
from MiniCPMO45.utils import TTSSamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="MiniCPM-O-4.5 Unified 语音-文本推理")
    parser.add_argument("--model-path", type=str, required=True,
                        help="MiniCPMO45 模型目录路径（含 config.json 与权重）")
    parser.add_argument("--system-prompt", type=str,
                        default="You are a helpful assistant. You can accept audio and text input and output voice and text.",
                        help="系统提示词")
    parser.add_argument("--input-audio", type=str, required=True,
                        help="用户输入音频路径，要求 16kHz 单声道 WAV")
    parser.add_argument("--input-text", type=str, default="",
                        help="用户输入文本（可选，与音频一起传入）")
    parser.add_argument("--output-text", type=str, default="output.txt",
                        help="输出文本保存路径")
    parser.add_argument("--output-audio", type=str, default="output.wav",
                        help="输出音频保存路径（24kHz WAV）")
    parser.add_argument("--ref-audio", type=str, default=None,
                        help="TTS 参考音频路径（用于音色克隆，默认复用 input-audio）")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="LLM 最大生成 token 数")
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备，如 cuda / cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="模型权重量化类型")
    # TTS 采样参数
    parser.add_argument("--tts-temperature", type=float, default=0.8)
    parser.add_argument("--tts-top-p", type=float, default=0.85)
    parser.add_argument("--tts-top-k", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"[1/5] 正在加载模型: {args.model_path} (dtype={args.dtype}, device={args.device}) ...")

    # 加载模型
    model = MiniCPMO.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device,
    )
    model.eval()

    # 关键修正: 使用 unified 初始化入口，自动加载 Token2Wav，节省显存
    print("[2/5] 初始化 Unified 模式 (chat_vocoder=token2wav) ...")
    model.init_unified(
        pt_path=None,               # 如需加载微调后的 .pt 权重，在此指定路径
        chat_vocoder="token2wav",   # 使用轻量 Token2Wav，不加载 CosyVoice2
        preload_both_tts=False,     # 仅加载 streaming TTS，进一步节省显存
    )

    # 显式切换到 CHAT 模式（单轮非流式对话）
    model.set_mode(ProcessorMode.CHAT)
    print("      Unified 初始化完成，当前模式: CHAT")

    # 准备输入数据
    print(f"[3/5] 加载输入音频: {args.input_audio}")
    audio_input, sr = librosa.load(args.input_audio, sr=16000, mono=True)
    print(f"      音频时长: {len(audio_input)/16000:.2f}s, 采样率: {sr}")

    # 参考音色（默认复用输入音频）
    tts_ref_audio = None
    if args.ref_audio:
        tts_ref_audio, _ = librosa.load(args.ref_audio, sr=16000, mono=True)
        print(f"      使用指定参考音频: {args.ref_audio}")
    else:
        tts_ref_audio = audio_input
        print(f"      使用输入音频作为 TTS 参考音色")

    # 构建对话消息: content 为列表，可混排 np.ndarray(音频) 和 str(文本)
    user_content = [audio_input]
    if args.input_text:
        user_content.append(args.input_text)

    msgs = [
        {"role": "system", "content": [args.system_prompt]},
        {"role": "user", "content": user_content},
    ]

    # TTS 采样参数
    tts_sampling_params = TTSSamplingParams(
        temperature=args.tts_temperature,
        top_p=args.tts_top_p,
        top_k=args.tts_top_k,
    )

    # 推理
    print("[4/5] 开始推理 (chat + TTS) ...")
    with torch.no_grad():
        result = model.chat(
            msgs=msgs,
            generate_audio=True,          # 生成语音
            use_tts_template=True,        # 启用 TTS 模板 <|tts_bos|> ... <|tts_eos|>
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            tts_ref_audio=tts_ref_audio,  # 传入参考音色
            tts_sampling_params=tts_sampling_params,
        )

    # 解析返回值
    if isinstance(result, tuple):
        text, waveform = result
    else:
        text = result
        waveform = None

    # 保存结果
    print("[5/5] 保存结果 ...")
    with open(args.output_text, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"      文本已保存: {args.output_text}")
    print(f"      生成内容:\n{text}")

    if waveform is not None:
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        sf.write(args.output_audio, waveform, samplerate=24000)
        print(f"\n      音频已保存: {args.output_audio} ({len(waveform)/24000:.2f}s @ 24kHz)")
    else:
        print("\n      警告: 未生成音频，请检查 use_tts_template / generate_audio 是否开启")


if __name__ == "__main__":
    main()
