#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离线批量解码训练时保存的 debug_info
用法：
    python offline_decode_batch.py \
        --debug_dir ./minicpmo_tts_finetuned/debug_info \
        --model_dir openbmb/MiniCPM-o-4_5
"""

import os
import sys
import json
import argparse
import inspect
import numpy as np
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from vq_codec import VQCodec


def get_decode_kwargs(vq_codec: VQCodec, add_silence: bool = True) -> dict:
    """检查 vq_codec.decode 是否支持 add_silence_prefix 参数"""
    sig = inspect.signature(vq_codec.decode)
    kwargs = {}
    if "add_silence_prefix" in sig.parameters:
        kwargs["add_silence_prefix"] = add_silence
    return kwargs


def decode_step(vq_codec: VQCodec, step_dir: str, sample_rate: int = 24000) -> bool:
    step_name = os.path.basename(step_dir)
    meta_path = os.path.join(step_dir, "meta.json")
    pred_npy = os.path.join(step_dir, "pred_tokens.npy")
    gt_npy = os.path.join(step_dir, "gt_tokens.npy")

    if not os.path.exists(meta_path):
        print(f"[Skip] {step_name}: meta.json not found")
        return False

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print(f"\n{'='*60}")
    print(f"[{step_name}]")
    print(f"  pred_text: {meta.get('pred_text', 'N/A')[:80]}")
    print(f"  raw_text:  {meta.get('raw_text', 'N/A')[:80]}")
    print(f"  audio_path: {meta.get('audio_path', 'N/A')}")

    gt_tokens = np.load(gt_npy).astype(np.int64)
    pred_tokens = np.load(pred_npy).astype(np.int64) if os.path.exists(pred_npy) else None

    print(f"  GT tokens:   shape={gt_tokens.shape}, range=[{gt_tokens.min()}, {gt_tokens.max()}]")
    if pred_tokens is not None:
        print(f"  Pred tokens: shape={pred_tokens.shape}, range=[{pred_tokens.min()}, {pred_tokens.max()}]")
        # 防御性截断
        pred_tokens = np.clip(pred_tokens, 0, 625).astype(np.int64)

    audio_path = meta.get("audio_path")
    ref_path = audio_path if (audio_path and os.path.exists(audio_path)) else None
    decode_kwargs = get_decode_kwargs(vq_codec, add_silence=True)

    # 解码 GT（验证闭环）
    gt_wav_path = os.path.join(step_dir, "gt.wav")
    try:
        gt_wav = vq_codec.decode(gt_tokens, prompt_wav_path=ref_path, **decode_kwargs)
        sf.write(gt_wav_path, gt_wav, sample_rate)
        print(f"  ✓ GT saved: {gt_wav_path} ({len(gt_wav)} samples)")
    except Exception as e:
        print(f"  ✗ GT failed: {type(e).__name__}: {e}")
        # fallback：不带 ref
        try:
            gt_wav = vq_codec.decode(gt_tokens, prompt_wav_path=None, **decode_kwargs)
            sf.write(gt_wav_path, gt_wav, sample_rate)
            print(f"  ✓ GT saved (no ref): {gt_wav_path}")
        except Exception as e2:
            print(f"  ✗ GT no-ref also failed: {type(e2).__name__}: {e2}")

    # 解码 Pred
    if pred_tokens is not None:
        pred_wav_path = os.path.join(step_dir, "pred.wav")
        try:
            pred_wav = vq_codec.decode(pred_tokens, prompt_wav_path=ref_path, **decode_kwargs)
            sf.write(pred_wav_path, pred_wav, sample_rate)
            print(f"  ✓ Pred saved: {pred_wav_path} ({len(pred_wav)} samples)")
        except Exception as e:
            print(f"  ✗ Pred failed: {type(e).__name__}: {e}")
            try:
                pred_wav = vq_codec.decode(pred_tokens, prompt_wav_path=None, **decode_kwargs)
                sf.write(pred_wav_path, pred_wav, sample_rate)
                print(f"  ✓ Pred saved (no ref): {pred_wav_path}")
            except Exception as e2:
                print(f"  ✗ Pred no-ref also failed: {type(e2).__name__}: {e2}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_dir", required=True)
    parser.add_argument("--model_dir", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not os.path.exists(args.debug_dir):
        print(f"Error: debug_dir not found: {args.debug_dir}")
        return

    print(f"Loading VQCodec from {args.model_dir} ...")
    vq_codec = VQCodec(model_dir=args.model_dir, device=args.device)
    print("VQCodec ready\n")

    step_dirs = sorted([
        os.path.join(args.debug_dir, d)
        for d in os.listdir(args.debug_dir)
        if d.startswith("step_") and os.path.isdir(os.path.join(args.debug_dir, d))
    ], key=lambda x: int(os.path.basename(x).split("_")[1]))

    print(f"Found {len(step_dirs)} steps to decode")

    success = 0
    for step_dir in step_dirs:
        if decode_step(vq_codec, step_dir):
            success += 1

    print(f"\n{'='*60}")
    print(f"Done: {success}/{len(step_dirs)} steps decoded")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
