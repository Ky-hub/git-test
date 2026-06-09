#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VQ Codec for MiniCPM-o-4.5
编码器: s3tokenizer (speech_tokenizer_v2_25hz)
解码器: stepaudio2 Token2wav
"""

import os
import argparse
import numpy as np
import soundfile as sf
import torch
from typing import Union, Optional, Dict, Tuple, List

# ==================== 依赖检查 ====================

try:
    import s3tokenizer
except ImportError:
    raise ImportError("缺少 s3tokenizer。请安装: pip install s3tokenizer")

try:
    from stepaudio2 import Token2wav
except ImportError:
    raise ImportError("缺少 stepaudio2。请安装: pip install stepaudio2-minicpmo")

try:
    from huggingface_hub import snapshot_download
    _HAS_HF = True
except ImportError:
    _HAS_HF = False


# ==================== 工具函数 ====================

def _hf_download(repo_id: str, allow_patterns: Optional[List[str]] = None) -> str:
    if not _HAS_HF:
        raise ImportError("pip install huggingface_hub")
    kwargs = {"repo_id": repo_id}
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    return snapshot_download(**kwargs)


def _find_subdir(base_dir: Optional[str], candidates: List[str]) -> Optional[str]:
    if not base_dir or not os.path.exists(base_dir):
        return None
    for c in candidates:
        p = os.path.join(base_dir, c)
        if os.path.exists(p):
            return p
    return None


# ==================== 核心类 ====================

class VQCodec:
    """
    VQ 音频编解码器
    - 编码: s3tokenizer (音频 -> VQ tokens)
    - 解码: stepaudio2 Token2wav (VQ tokens -> 24kHz 音频)
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        s3tokenizer_dir: Optional[str] = None,
        token2wav_dir: Optional[str] = None,
        device: str = "cuda",
        s3tokenizer_name: str = "speech_tokenizer_v2_25hz",
        float16: bool = False,
        n_timesteps: int = 5,
    ):
        """
        Args:
            model_dir: MiniCPM-o-4.5 根目录（自动找子目录）或 HF repo_id
            s3tokenizer_dir: s3tokenizer 模型目录（优先于 model_dir）
            token2wav_dir: Token2wav 模型目录（优先于 model_dir）
            device: cuda / cpu
            s3tokenizer_name: s3tokenizer 模型名，MiniCPM-o 使用 v2_25hz
            float16: Token2wav 是否使用 FP16
            n_timesteps: Token2wav 流式解码步数
        """
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.input_sr = 16000
        self.output_sr = 24000

        # 路径解析：如果传入 HF repo_id 或不存在路径，自动下载
        base_dir = model_dir
        if base_dir and not os.path.exists(base_dir):
            base_dir = _hf_download(base_dir)

        # 初始化编码器 & 解码器
        self.encoder = self._init_s3tokenizer(s3tokenizer_dir or base_dir, s3tokenizer_name)
        self.decoder = self._init_token2wav(token2wav_dir or base_dir, float16, n_timesteps)

    def _init_s3tokenizer(self, model_dir: Optional[str], model_name: str):
        """加载 s3tokenizer 编码器"""
        if model_dir and os.path.exists(model_dir):
            files = os.listdir(model_dir) if os.path.isdir(model_dir) else []
            if any(f in files for f in ["config.json", "pytorch_model.bin", "model.safetensors"]):
                print(f"[VQCodec] S3Tokenizer from local: {model_dir}")
                try:
                    tokenizer = s3tokenizer.load_model(model_dir).to(self.device)
                    tokenizer.eval()
                    return tokenizer
                except Exception as e:
                    print(f"      本地加载失败: {e}，尝试用模型名加载")

        print(f"[VQCodec] S3Tokenizer loading: {model_name}")
        tokenizer = s3tokenizer.load_model(model_name).to(self.device)
        tokenizer.eval()
        return tokenizer

    def _init_token2wav(self, model_dir: Optional[str], float16: bool, n_timesteps: int):
        """加载 Token2wav 解码器"""
        candidates = [
            model_dir,
            _find_subdir(model_dir, ["assets/token2wav", "token2wav"]),
        ]
        decoder_dir = None
        for p in candidates:
            if p and os.path.exists(p):
                decoder_dir = p
                break

        if decoder_dir is None:
            print("[VQCodec] 下载 Token2wav...")
            cache = _hf_download("openbmb/MiniCPM-o-4_5", allow_patterns=["assets/token2wav/**"])
            decoder_dir = os.path.join(cache, "assets", "token2wav")

        print(f"[VQCodec] Token2wav: {decoder_dir}")
        return Token2wav(decoder_dir, float16=float16, n_timesteps=n_timesteps)

    # ==================== 编码 ====================

    def encode(self, audio_input: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        音频 -> VQ Tokens

        Args:
            audio_input: 音频路径(16kHz wav) / numpy 数组 / tensor

        Returns:
            VQ token IDs, shape (num_tokens,)
        """
        if isinstance(audio_input, str):
            audio = s3tokenizer.load_audio(audio_input)
        elif isinstance(audio_input, torch.Tensor):
            audio = audio_input.cpu()
        else:
            audio = torch.from_numpy(np.array(audio_input))

        # 确保一维
        if audio.dim() > 1:
            audio = audio.mean(dim=-1) if audio.dim() > 1 else audio.mean()
        audio = audio.squeeze().float()

        # 官方流程: log_mel_spectrogram -> padding -> quantize
        mel = s3tokenizer.log_mel_spectrogram(audio)  # (80, T)
        mels, mels_lens = s3tokenizer.padding([mel])
        mels = mels.to(self.device)
        mels_lens = mels_lens.to(self.device)

        with torch.no_grad():
            codes, codes_lens = self.encoder.quantize(mels, mels_lens)

        valid_len = codes_lens[0].item()
        tokens = codes[0, :valid_len].cpu().numpy()
        return tokens

    # ==================== 解码 ====================

    def decode(
        self,
        tokens: Union[np.ndarray, torch.Tensor, list],
        prompt_wav_path: Optional[str] = None,
        add_silence_prefix: bool = True,
    ) -> np.ndarray:
        """
        VQ Tokens -> 音频波形 (24kHz)

        Args:
            tokens: VQ token IDs
            prompt_wav_path: 参考音频路径（用于音色克隆，强烈建议传入原始音频）
            add_silence_prefix: 是否在 token 前添加 silence token (4218*3)，与 MiniCPM-o 行为一致

        Returns:
            音频波形, np.ndarray, float32, 24kHz
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        token_list = np.array(tokens).reshape(-1).tolist()

        if not token_list:
            return np.array([], dtype=np.float32)

        # MiniCPM-o 原生行为：解码前加 3 个 silence token (4218)
        if add_silence_prefix:
            token_list = [4218] * 3 + token_list

        # 设置参考音频缓存（保持音色一致）
        if prompt_wav_path and os.path.exists(prompt_wav_path):
            self.decoder.cache = None
            flow_cache, hift_cache = self.decoder.set_stream_cache(prompt_wav_path)
            self.decoder.stream_cache = flow_cache
            self.decoder.hift_cache_dict = hift_cache

        # 分块解码（防止长序列溢出）
        CHUNK_SIZE = 25
        pre_lookahead = 3
        if hasattr(self.decoder, "flow") and hasattr(self.decoder.flow, "pre_lookahead_len"):
            pre_lookahead = int(self.decoder.flow.pre_lookahead_len)

        chunks = []
        pos = 0
        total = len(token_list)

        while pos + CHUNK_SIZE + pre_lookahead <= total:
            chunk = token_list[pos : pos + CHUNK_SIZE + pre_lookahead]
            is_last = (pos + CHUNK_SIZE + pre_lookahead >= total)
            wav = self.decoder.stream(
                chunk,
                prompt_wav=prompt_wav_path,
                last_chunk=is_last,
                return_waveform=True,
            )
            if wav is not None:
                chunks.append(np.array(wav).squeeze())
            pos += CHUNK_SIZE

        # flush 剩余
        if pos < total:
            wav = self.decoder.stream(
                token_list[pos:],
                prompt_wav=prompt_wav_path,
                last_chunk=True,
                return_waveform=True,
            )
            if wav is not None:
                chunks.append(np.array(wav).squeeze())

        return np.concatenate(chunks).astype(np.float32) if chunks else np.array([], dtype=np.float32)

    def decode_to_file(
        self,
        tokens: Union[np.ndarray, list],
        output_path: str,
        prompt_wav_path: Optional[str] = None,
        add_silence_prefix: bool = True,
    ) -> str:
        """解码并保存为 wav"""
        waveform = self.decode(tokens, prompt_wav_path, add_silence_prefix)
        sf.write(output_path, waveform, self.output_sr)
        return output_path

    # ==================== 便捷方法 ====================

    def roundtrip(
        self,
        audio_input: Union[str, np.ndarray],
        prompt_wav_path: Optional[str] = None,
        add_silence_prefix: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        闭环测试: 音频 -> VQ -> 音频

        Returns:
            (original_audio, tokens, reconstructed_audio)
        """
        tokens = self.encode(audio_input)
        recon = self.decode(
            tokens,
            prompt_wav_path=prompt_wav_path or (audio_input if isinstance(audio_input, str) else None),
            add_silence_prefix=add_silence_prefix,
        )
        if isinstance(audio_input, str):
            orig, sr = sf.read(audio_input)
            if sr != self.input_sr:
                raise ValueError(f"要求 16kHz, 当前 {sr}Hz")
        else:
            orig = np.array(audio_input)
        if orig.ndim > 1:
            orig = orig.mean(axis=-1)
        return orig.astype(np.float32), tokens, recon


# ==================== 测试函数 ====================

def test_codec(
    audio_path: str,
    model_dir: Optional[str] = None,
    s3tokenizer_name: str = "speech_tokenizer_v2_25hz",
    output_dir: str = "./vq_test_output",
    add_silence_prefix: bool = True,
) -> Dict:
    """
    闭环测试：编码 -> 解码 -> 对比

    输出文件:
        - original.wav
        - reconstructed.wav
        - vq_tokens.npy
        - report.txt
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("VQ Codec 闭环测试")
    print("=" * 60)

    print("\n[1/4] 初始化 VQCodec...")
    codec = VQCodec(model_dir=model_dir, s3tokenizer_name=s3tokenizer_name)

    print(f"\n[2/4] 加载并编码: {audio_path}")
    orig, sr = sf.read(audio_path)
    if sr != 16000:
        raise ValueError(f"要求 16kHz wav, 当前 {sr}Hz")
    print(f"      原始音频: {len(orig)/16000:.2f}s @ 16kHz")
    sf.write(os.path.join(output_dir, "original.wav"), orig, 16000)

    tokens = codec.encode(audio_path)
    print(f"      VQ Tokens: {tokens.shape}, range [{np.min(tokens)}, {np.max(tokens)}]")
    np.save(os.path.join(output_dir, "vq_tokens.npy"), tokens)

    print("\n[3/4] 解码 -> 音频...")
    recon = codec.decode(tokens, prompt_wav_path=audio_path, add_silence_prefix=add_silence_prefix)
    print(f"      重构音频: {len(recon)} samples @ 24kHz")
    sf.write(os.path.join(output_dir, "reconstructed.wav"), recon, 24000)

    print("\n[4/4] 对比分析...")
    metrics = {}
    try:
        from scipy import signal
        target_len = int(len(orig) * 24000 / 16000)
        ref = signal.resample(orig, target_len)[: len(recon)]
        hyp = recon[: len(ref)]

        mse = np.mean((ref - hyp) ** 2)
        snr = 10 * np.log10(np.mean(ref**2) / (mse + 1e-10))
        corr = np.corrcoef(ref, hyp)[0, 1]

        metrics = {"mse": mse, "snr": snr, "corr": corr}
        print(f"      MSE: {mse:.6f} | SNR: {snr:.2f} dB | CORR: {corr:.4f}")
    except ImportError:
        print("      [跳过] 未安装 scipy，无法计算对比指标")

    # 保存报告
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("VQ Codec 闭环测试报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"原始音频: {audio_path}\n")
        f.write(f"Token shape: {tokens.shape}\n")
        f.write(f"Token range: [{np.min(tokens)}, {np.max(tokens)}]\n")
        if metrics:
            f.write(f"MSE: {metrics['mse']:.6f}\n")
            f.write(f"SNR: {metrics['snr']:.2f} dB\n")
            f.write(f"CORR: {metrics['corr']:.4f}\n")

    print(f"\n{'='*60}")
    print(f"完成。输出目录: {output_dir}")
    print(f"{'='*60}")

    return {
        "original": orig,
        "tokens": tokens,
        "reconstructed": recon,
        "metrics": metrics,
        "output_dir": output_dir,
    }


# ==================== CLI 入口 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQ Codec 编解码测试")
    parser.add_argument("--audio", type=str, required=True, help="输入音频路径 (16kHz wav)")
    parser.add_argument("--model_dir", type=str, default=None, help="MiniCPM-o-4.5 根目录 或 HF repo_id")
    parser.add_argument("--s3tokenizer_name", type=str, default="speech_tokenizer_v2_25hz", help="s3tokenizer 模型名")
    parser.add_argument("--output_dir", type=str, default="./vq_test_output", help="输出目录")
    parser.add_argument("--no_silence_prefix", action="store_true", help="不加 silence token 前缀")
    args = parser.parse_args()

    test_codec(
        args.audio,
        model_dir=args.model_dir,
        s3tokenizer_name=args.s3tokenizer_name,
        output_dir=args.output_dir,
        add_silence_prefix=not args.no_silence_prefix,
    )
