#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_evaluation_local.py

MiniCPM-o-4.5 本地模型评估脚本
==============================

直接加载 MiniCPMO 模型进行推理（参考 infer_debug.py 模式），
绕过 Gateway/Worker 网络服务，完全避免 403/连接问题。

适用场景：
- 评估服务器有 GPU，可直接加载模型
- 内网 Gateway/Worker 有连接问题
- 需要稳定的批量评估

要求：
- 与 MiniCPMO45 代码同目录（或设置 PYTHONPATH）
- GPU 显存足够加载模型

启动方式（与 infer_debug.py 一致）：
    cd /path/to/Train
    PYTHONPATH=. python adapters/run_evaluation_local.py \
        --model-path /path/to/base_model \
        --data-dir /path/to/fdb_v3_data_released \
        --output-dir ./eval_results
"""

import os
import sys
import json
import time
import base64
import logging
import argparse
import tempfile
import datetime
import statistics
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict

import numpy as np
import librosa
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_local")

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_JSON_PATH = PROJECT_ROOT / "benchmark_data_v2.json"

# 数据目录：fdb_v3_data_released 默认与 benchmark_data_v2.json 同级
DEFAULT_DATA_DIR = PROJECT_ROOT / "fdb_v3_data_released"

# 内部 LLM 裁判
INTERNAL_LLM_URL = os.environ.get("INTERNAL_LLM_URL", "")
INTERNAL_LLM_MODEL = os.environ.get("INTERNAL_LLM_MODEL", "qwen2.5-72b")


# =============================================================================
# Data Loading
# =============================================================================

def load_benchmark_data(data_json_path: str):
    """加载 benchmark_data_v2.json"""
    if not os.path.exists(data_json_path):
        logger.warning(f"Data JSON not found: {data_json_path}")
        return {}
    with open(data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict) and "scenarios" in data:
            data = data["scenarios"]
    return {item["id"]: item for item in data}


def discover_inputs(root_dir: str):
    """发现所有 input.wav / input.mp4 文件"""
    root = Path(root_dir)
    inputs = []
    if not root.exists():
        return inputs

    def pick_input(ex_dir: Path):
        """优先选 input.mp4（音视频），其次 input.wav（纯音频）"""
        mp4 = ex_dir / "input.mp4"
        if mp4.exists():
            return str(mp4)
        wav = ex_dir / "input.wav"
        if wav.exists():
            return str(wav)
        return None

    # 嵌套: {pid}/example_{id}/input.{wav,mp4}
    for pid_dir in sorted(root.iterdir()):
        if not pid_dir.is_dir() or pid_dir.name.startswith("."):
            continue
        pid = pid_dir.name
        for ex_dir in sorted(pid_dir.iterdir()):
            if not ex_dir.is_dir() or not ex_dir.name.startswith("example_"):
                continue
            eid = ex_dir.name.replace("example_", "")
            inp = pick_input(ex_dir)
            if inp:
                inputs.append((pid, eid, inp))

    # 扁平: {id}_{pid}/input.{wav,mp4}
    if not inputs:
        import re
        pat = re.compile(r"^(.+)_([0-9a-f]{24})$")
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            m = pat.match(folder.name)
            if m:
                eid, pid = m.group(1), m.group(2)
                inp = pick_input(folder)
                if inp:
                    inputs.append((pid, eid, inp))

    wav_count = sum(1 for _, _, p in inputs if p.endswith(".wav"))
    mp4_count = sum(1 for _, _, p in inputs if p.endswith(".mp4"))
    logger.info(f"Found {len(inputs)} inputs: {wav_count} audio, {mp4_count} video")
    return inputs


# =============================================================================
# LLM Judge
# =============================================================================

class LLMJudge:
    def __init__(self):
        self.url = INTERNAL_LLM_URL
        self.model = INTERNAL_LLM_MODEL
        if not self.url:
            self.client = None
            return
        import httpx
        self.client = httpx.Client(timeout=httpx.Timeout(30.0))

    def evaluate(self, query: str, response: str) -> Dict:
        if not self.client:
            return self._fallback(response)
        system = "你是语音交互质量评估专家。评分（1-5分）：Naturalness, Relevance, Completeness, Fluency, Overall, Reasoning"
        user = f"用户问题: {query}\n\nAI回答: {response}\n\n请评分。"
        try:
            r = self.client.post(self.url, json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.1, "max_tokens": 256,
            })
            r.raise_for_status()
            raw_json = r.json()
            content = raw_json.get("choices", [{}])[0].get("message", {}).get("content")
            if not content:
                logger.warning(f"Judge returned empty content. Raw: {raw_json}")
                return self._fallback(response)
            logger.debug(f"Judge raw response: {content[:200]}")
            return self._parse(content)
        except Exception as e:
            logger.warning(f"Judge failed: {e}")
            return self._fallback(response)

    def _parse(self, text: str) -> Dict:
        if text is None:
            return self._fallback("")
        r = {"naturalness": 3, "relevance": 3, "completeness": 3, "fluency": 3, "overall": 3, "reasoning": ""}
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("Naturalness:"): r["naturalness"] = self._s(line)
            elif line.startswith("Relevance:"): r["relevance"] = self._s(line)
            elif line.startswith("Completeness:"): r["completeness"] = self._s(line)
            elif line.startswith("Fluency:"): r["fluency"] = self._s(line)
            elif line.startswith("Overall:"): r["overall"] = self._s(line)
            elif line.startswith("Reasoning:"): r["reasoning"] = line[10:].strip()
        return r

    @staticmethod
    def _s(line: str) -> int:
        try:
            parts = line.split(":", 1)  # 只分割第一个冒号
            if len(parts) < 2:
                return 3
            val = parts[1].strip()
            if not val:
                return 3
            return int(val.split()[0])
        except Exception:
            return 3

    def _fallback(self, text: str) -> Dict:
        score = 4 if len(text) > 20 else 3
        return {"naturalness": score, "relevance": score, "completeness": score,
                "fluency": score, "overall": score, "reasoning": "fallback"}

    def close(self):
        if self.client:
            self.client.close()


# =============================================================================
# 模型推理（核心，参考 infer_debug.py）
# =============================================================================

def _import_minicpmo45():
    """
    兼容多种 MiniCPMO45 目录位置的导入方式
    
    尝试顺序：
    1. PYTHONPATH 中的 MiniCPMO45
    2. 与当前脚本同级目录下的 MiniCPMO45
    3. 项目根目录下的 MiniCPMO45
    """
    import importlib.util

    # 尝试直接导入（PYTHONPATH 已设置时）
    try:
        from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO
        from MiniCPMO45.utils import TTSSamplingParams
        return MiniCPMO, TTSSamplingParams
    except ImportError:
        pass

    # 尝试从脚本所在目录的父目录导入
    script_dir = Path(__file__).resolve().parent
    for base in [script_dir, script_dir.parent]:
        init_file = base / "MiniCPMO45" / "__init__.py"
        model_file = base / "MiniCPMO45" / "modeling_minicpmo_unified.py"
        if init_file.exists() and model_file.exists():
            spec = importlib.util.spec_from_file_location(
                "MiniCPMO45", str(init_file),
                submodule_search_locations=[str(base / "MiniCPMO45")]
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules["MiniCPMO45"] = mod
            spec.loader.exec_module(mod)
            # 再导入子模块
            model_module = importlib.import_module("MiniCPMO45.modeling_minicpmo_unified")
            utils_module = importlib.import_module("MiniCPMO45.utils")
            return model_module.MiniCPMO, utils_module.TTSSamplingParams

    # 都失败了，给出清晰的错误提示
    raise ImportError(
        "无法导入 MiniCPMO45。请确保以下任一条件满足：\n"
        "1. 设置 PYTHONPATH 包含 MiniCPMO45 的父目录：\n"
        "   export PYTHONPATH=/path/to/Train:$PYTHONPATH\n"
        "2. 将 MiniCPMO45 目录放在以下位置之一：\n"
        f"   - {script_dir}/MiniCPMO45/\n"
        f"   - {script_dir.parent}/MiniCPMO45/\n"
        "3. 或者直接 cd 到 Train 目录运行：\n"
        "   cd /path/to/Train && PYTHONPATH=. python adapters/run_evaluation_local.py ..."
    )


class MiniCPMOLocalInference:
    """
    本地模型推理封装
    
    与 infer_debug.py 的半双工模式一致：
      model.chat(msgs, generate_audio=True, use_tts_template=True, ...)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 512,
    ):
        import torch

        MiniCPMO, TTSSamplingParams = _import_minicpmo45()

        self.max_new_tokens = max_new_tokens
        self.tts_params = TTSSamplingParams(temperature=0.8, top_p=0.85, top_k=25)

        logger.info(f"Loading model from {model_path}...")

        # 加载模型
        torch_dtype = getattr(torch, dtype)
        self.model = MiniCPMO.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.model.eval()

        # 初始化 Token2Wav（streaming 模式，与 infer_debug.py 一致）
        # 注意：不调用 init_unified()，避免 DuplexCapability 修改模型状态
        self.model.init_token2wav(streaming=True)

        # processor 是延迟初始化的（第一次 chat() 时才创建）
        # 这里先不获取 tokenizer，需要时通过 property 延迟加载
        self._tokenizer = None
        logger.info("Model loaded successfully")

    @property
    def tokenizer(self):
        """延迟获取 tokenizer（在第一次 chat() 后 processor 才被创建）"""
        if self._tokenizer is None:
            # 触发 processor 初始化
            if not hasattr(self.model, 'processor') or self.model.processor is None:
                from MiniCPMO45.processing_minicpmo import MiniCPMOProcessor
                self.model.processor = MiniCPMOProcessor.from_pretrained(
                    self.model.config._name_or_path, trust_remote_code=True
                )
            self._tokenizer = self.model.processor.tokenizer
        return self._tokenizer

    def infer(
        self,
        audio_path: str,
        video_path: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant.",
    ):
        """
        对单个样本进行推理（支持纯音频或音视频）
        与 infer_debug.py 一致：使用 model.chat() 非流式半双工模式

        Args:
            audio_path: 输入音频路径（用于 TTS 参考音色）
            video_path: 输入视频路径（可选）
            system_prompt: 系统提示词

        Returns:
            dict: {
                "text": str,               # 回复文本
                "waveform": np.array|None, # 输出音频波形
                "latency_ms": float,       # 总延迟
                "audio_duration_ms": float,# 输入音频时长
            }
        """
        import torch

        # 加载音频
        audio_input, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_duration_ms = len(audio_input) / sr * 1000

        # 构建消息（与源码 chat() 对齐）
        if video_path and os.path.exists(video_path):
            user_content = [video_path, audio_input]
            logger.debug(f"Video+Audio mode: {video_path}")
        else:
            user_content = [audio_input]

        msgs = [
            {"role": "system", "content": [system_prompt]},
            {"role": "user", "content": user_content},
        ]

        # 截断参考音频避免 Token2Wav cache 溢出
        # Token2Wav 的 set_stream_cache 对长音频可能溢出
        max_ref_sec = 10  # 最多取 10 秒作为参考音色
        max_ref_samples = max_ref_sec * sr
        if len(audio_input) > max_ref_samples:
            tts_ref = audio_input[:max_ref_samples]
            logger.debug(f"Truncated ref audio: {len(audio_input)} -> {max_ref_samples} samples ({max_ref_sec}s)")
        else:
            tts_ref = audio_input

        # 推理（非流式）
        start = time.perf_counter()

        result = self.model.chat(
            msgs=msgs,
            generate_audio=True,
            use_tts_template=True,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            tts_ref_audio=tts_ref,
            tts_sampling_params=self.tts_params,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # 解析结果：chat() 返回 (text, waveform) 或 text
        if isinstance(result, tuple):
            text, waveform = result
        else:
            text = result
            waveform = None

        # 波形转 numpy
        if waveform is not None and isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()

        return {
            "text": text or "",
            "waveform": waveform,
            "latency_ms": elapsed_ms,
            "audio_duration_ms": audio_duration_ms,
        }


# =============================================================================
# 评估流程
# =============================================================================

def save_wav(waveform, path: str, sr: int = 24000):
    """保存音频"""
    if waveform is None:
        return
    sf.write(path, waveform, samplerate=sr)


def process_single(
    pid: str, example_id: str, input_path: str,
    model: MiniCPMOLocalInference,
    data: Dict, judge: Optional[LLMJudge],
    output_dir: str,
    provider: str = "minicpmo45_local",
    force: bool = False,
) -> Optional[Dict]:
    """处理单个样本 — 参照 run_tool_benchmark.py 的流程"""
    item = data.get(example_id)
    if item is None:
        logger.warning(f"Example {example_id} not found in data — skipping")
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"output_{provider}_{pid}_{example_id}.wav"
    result_path = out_dir / f"result_{provider}_{pid}_{example_id}.json"

    # 检查是否已评估
    if result_path.exists() and not force:
        logger.info(f"Already evaluated — skipping (use --force to re-run)")
        return None

    category = item.get("domain", item.get("category", "unknown"))
    result = {
        "pid": pid,
        "example_id": example_id,
        "category": category,
        "title": item.get("title", ""),
        "provider": provider,
        "evaluated_at": datetime.datetime.now().isoformat(),
    }

    # 判断输入类型
    is_video = str(input_path).endswith(".mp4")
    audio_path = input_path
    video_path = input_path if is_video else None
    result["input_type"] = "video" if is_video else "audio"

    # Step 1: 模型推理
    logger.info(f"Evaluating: {example_id} [{category}] type={result['input_type']}")
    inference_start = time.time()
    try:
        system_prompt = item.get("system_prompt", "You are a helpful assistant.")
        infer_result = model.infer(
            audio_path=audio_path,
            video_path=video_path,
            system_prompt=system_prompt,
        )
        inference_time = time.time() - inference_start
        result["inference_time_s"] = round(inference_time, 2)

        text = infer_result["text"]
        waveform = infer_result["waveform"]

        result["response_text"] = text
        result["latency_ms"] = round(infer_result["latency_ms"], 1)

        # 保存音频
        if waveform is not None:
            save_wav(waveform, str(output_path))
            result["output_audio"] = str(output_path)

        # 计算 RTF
        audio_duration_ms = infer_result["audio_duration_ms"]
        result["input_audio_duration_ms"] = round(audio_duration_ms, 1)
        if audio_duration_ms > 0:
            result["rtf"] = round(infer_result["latency_ms"] / audio_duration_ms, 3)

        logger.info(f"  Text: {text[:100]}...")
        logger.info(f"  E2E: {result['latency_ms']:.0f}ms")

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        result["status"] = "inference_error"
        result["error"] = str(e)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return result

    # Step 2: LLM 裁判（评估人性化，可选）
    if judge and text:
        # 使用 title 作为 query 的替代（与原始脚本一致，data 中没有 query 字段）
        query_text = item.get("title", "")
        if query_text:
            result["human_likeness"] = judge.evaluate(query_text, text)

    result["status"] = "completed"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved: {result_path.name}")

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MiniCPM-o-4.5 Local Evaluation")
    parser.add_argument("--model-path", required=True, help="模型路径（含 config.json）")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="fdb_v3_data_released 目录")
    parser.add_argument("--output-dir", default="./eval_results", help="输出目录")
    parser.add_argument("--provider", default="minicpmo45_local")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--pid", help="只处理指定 participant ID")
    parser.add_argument("--example", help="只处理指定 example ID")
    parser.add_argument("--force", action="store_true", help="覆盖已有结果")
    parser.add_argument("--data-json", default=str(DATA_JSON_PATH))

    args = parser.parse_args()

    # 加载模型
    logger.info("=" * 60)
    logger.info(f"Initializing local model: {args.provider}")
    logger.info("=" * 60)

    model = MiniCPMOLocalInference(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )

    # 加载数据
    data = load_benchmark_data(args.data_json)
    logger.info(f"Loaded {len(data)} scenarios")

    # 发现输入
    inputs = discover_inputs(args.data_dir)
    logger.info(f"Found {len(inputs)} input files")

    if not inputs:
        logger.error("No input files found. Check that fdb_v3_data_released/ exists.")
        sys.exit(1)

    # 过滤
    if args.pid:
        inputs = [(p, e, i) for p, e, i in inputs if p == args.pid]
        logger.info(f"Filtered to PID={args.pid}: {len(inputs)}")
    if args.example:
        inputs = [(p, e, i) for p, e, i in inputs if e == args.example]
        logger.info(f"Filtered to example={args.example}: {len(inputs)}")

    # Judge
    judge = LLMJudge() if INTERNAL_LLM_URL else None

    # 处理所有样本
    success = failed = skipped = 0
    all_results = []

    for pid, example_id, input_path in inputs:
        item = data.get(example_id)
        category = item.get("domain", item.get("category", "unknown")) if item else "unknown"
        pid_short = pid[:8]
        logger.info(f"\n{'='*60}")
        logger.info(f"PID={pid_short}... / example_{example_id} [{category}]")

        result = process_single(
            pid, example_id, input_path, model,
            data, judge, args.output_dir,
            args.provider, args.force,
        )

        if result is None:
            skipped += 1
        elif result.get("status") == "completed":
            all_results.append(result)
            success += 1
        else:
            all_results.append(result)
            failed += 1

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation Summary ({args.provider})")
    logger.info(f"Completed: {success}, Failed: {failed}, Skipped: {skipped}")

    completed = [r for r in all_results if r.get("status") == "completed"]
    if completed:
        latencies = [r["latency_ms"] for r in completed]
        logger.info(f"Avg E2E latency: {statistics.mean(latencies):.0f}ms")

        rtf_vals = [r["rtf"] for r in completed if "rtf" in r]
        if rtf_vals:
            logger.info(f"Avg RTF: {statistics.mean(rtf_vals):.3f}")

        if judge:
            scores = [r["human_likeness"]["overall"] for r in completed if "human_likeness" in r]
            if scores:
                logger.info(f"Avg human-likeness: {statistics.mean(scores):.1f}/5")

    # Save aggregate summary
    summary_path = Path(args.output_dir) / f"evaluation_summary_{args.provider}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "provider": args.provider,
            "evaluated_at": datetime.datetime.now().isoformat(),
            "total": len(all_results),
            "completed": success,
            "failed": failed,
            "skipped": skipped,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved: {summary_path}")

    if judge:
        judge.close()


if __name__ == "__main__":
    main()
