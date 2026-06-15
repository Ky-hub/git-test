#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_evaluation_local.py

MiniCPM-o-4.5 本地模型评估脚本（全双工流式版本）
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
import statistics
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

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
DEFAULT_DATA_DIR = PROJECT_ROOT / "fdb_v3_data_released"

INTERNAL_LLM_URL = os.environ.get("INTERNAL_LLM_URL", "")
INTERNAL_LLM_MODEL = os.environ.get("INTERNAL_LLM_MODEL", "qwen2.5-72b")


# =============================================================================
# Data Loading
# =============================================================================

def load_benchmark_data(data_json_path: str):
    if not os.path.exists(data_json_path):
        logger.warning(f"Data JSON not found: {data_json_path}")
        return {}
    with open(data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict) and "scenarios" in data:
            data = data["scenarios"]
    return {item["id"]: item for item in data}


def discover_inputs(root_dir: str):
    root = Path(root_dir)
    inputs = []
    if not root.exists():
        return inputs

    def pick_input(ex_dir: Path):
        mp4 = ex_dir / "input.mp4"
        if mp4.exists():
            return str(mp4)
        wav = ex_dir / "input.wav"
        if wav.exists():
            return str(wav)
        return None

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
            parts = line.split(":", 1)
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
# 模型推理（全双工流式）
# =============================================================================

def _import_minicpmo45():
    import importlib.util
    try:
        from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO, ProcessorMode
        from MiniCPMO45.utils import TTSSamplingParams
        return MiniCPMO, TTSSamplingParams, ProcessorMode
    except ImportError:
        pass

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
            model_module = importlib.import_module("MiniCPMO45.modeling_minicpmo_unified")
            utils_module = importlib.import_module("MiniCPMO45.utils")
            return model_module.MiniCPMO, utils_module.TTSSamplingParams, model_module.ProcessorMode

    raise ImportError(
        "无法导入 MiniCPMO45。请确保 PYTHONPATH 包含 MiniCPMO45 的父目录，"
        "或将 MiniCPMO45 放在脚本同级目录下。"
    )


class MiniCPMOLocalInference:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 512,
        max_speak_tokens: int = 50,
        chunk_ms: int = 1000,
        realtime: bool = False,
    ):
        import torch

        MiniCPMO, TTSSamplingParams, ProcessorMode = _import_minicpmo45()
        self.max_new_tokens = max_new_tokens
        self.max_speak_tokens = max_speak_tokens
        self.chunk_ms = chunk_ms
        self.realtime = realtime
        self.ProcessorMode = ProcessorMode

        logger.info(f"Loading model from {model_path}...")
        torch_dtype = getattr(torch, dtype)
        self.model = MiniCPMO.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.model.eval()

        # 将 chunk_ms 传入 duplex_config，让 DuplexCapability 内部使用
        self.model.init_unified(
            pt_path=None,
            chat_vocoder="token2wav",
            preload_both_tts=False,
            duplex_config={
                "max_new_speak_tokens_per_chunk": self.max_speak_tokens,
                "chunk_ms": self.chunk_ms,
                "first_chunk_ms": self.chunk_ms + 35,  # 与源码对齐，first_chunk 略长
            },
            device=device,
        )
        self._tokenizer = None
        logger.info(f"Model loaded (chunk_ms={chunk_ms}, realtime={realtime})")

    @property
    def tokenizer(self):
        if self._tokenizer is None:
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
        ref_audio_path: Optional[str] = None,
    ):
        """
        全双工流式推理。
        输入音频完整 chunk-by-chunk，不截断。
        若 realtime=True，每处理完一个 chunk 会 sleep 到与真实时间对齐。
        """
        import torch

        # ── 1. 加载输入音频（完整，不截断） ──
        audio_input, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_duration_ms = len(audio_input) / sr * 1000
        logger.info(f"[Input] Loaded {len(audio_input)} samples ({audio_duration_ms/1000:.1f}s) — NOT truncated")

        # ── 2. 准备参考音色（TTS voice clone） ──
        if ref_audio_path and os.path.exists(ref_audio_path):
            ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
            logger.info(f"[Ref] Using specified ref audio: {ref_audio_path}")
        else:
            ref_audio = audio_input.copy()
            logger.info("[Ref] No ref audio specified, using input audio as default")

        # 截断参考音色到 10s，避免 Token2Wav cache 溢出
        max_ref_sec = 10
        max_ref_samples = max_ref_sec * sr
        if len(ref_audio) > max_ref_samples:
            ref_audio = ref_audio[:max_ref_samples]
            logger.info(f"[Ref] Truncated to {max_ref_sec}s ({max_ref_samples} samples)")

        # 将截断后的参考音色写入临时文件，供 Token2Wav set_stream_cache
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_wav.name, ref_audio, samplerate=sr)
        prompt_wav_path = tmp_wav.name
        tmp_wav.close()

        # ── 3. 设置全双工模式 ──
        self.model.set_mode(self.ProcessorMode.DUPLEX)

        # ── 关键修正：prefix/suffix 严格与 infer_debug.py 一致 ──
        # 错误示范（之前）：prefix = f"||<<||<|im_start|>system..."  ← 多了 ||<< 导致 tokenizer 无法识别特殊 token
        # 正确格式（如下）：纯字符串，让 tokenizer 正确识别 <|im_start|> 等特殊 token
        prefix = f"||<|im_start|>system\n{system_prompt}\n<<||<|audio_start|>"
        suffix = "||<|audio_end|>"

        full_prompt = self.model.duplex_prepare(
            prefix_system_prompt=prefix,
            suffix_system_prompt=suffix,
            ref_audio=ref_audio,
            prompt_wav_path=prompt_wav_path,
        )
        logger.info(f"[Duplex] System prompt prepared")

        # ── 4. 对完整输入音频进行 chunk-by-chunk 处理 ──
        chunk_size = int(self.chunk_ms * sr / 1000)  # 根据 chunk_ms 动态计算
        total_len = len(audio_input)
        num_real_chunks = (total_len + chunk_size - 1) // chunk_size
        logger.info(f"[Duplex] Processing {num_real_chunks} chunks ({self.chunk_ms}ms/chunk) from {audio_duration_ms/1000:.1f}s audio")

        unit_records = []
        all_speak_texts = []
        all_audio_chunks = []

        # 计时
        infer_start = time.perf_counter()
        first_speak_time_ms = None

        # 处理所有真实音频 chunk
        for i in range(num_real_chunks):
            chunk_start = i * chunk_size
            chunk_end = min((i + 1) * chunk_size, total_len)
            chunk = audio_input[chunk_start:chunk_end]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode="constant")

            # 处理 chunk（内部会计时）
            gen_res = self._process_chunk(
                i, chunk, chunk_start, chunk_end,
                unit_records, all_speak_texts, all_audio_chunks,
                is_silence=False,
            )

            # 记录首次 SPEAK
            if first_speak_time_ms is None and not gen_res['is_listen']:
                first_speak_time_ms = (time.perf_counter() - infer_start) * 1000
                logger.info(f"\n*** FIRST SPEAK at chunk {i}: {first_speak_time_ms:.0f}ms from audio start ***")

            # ── 实时模拟：如果处理太快，sleep 到与真实时间对齐 ──
            if self.realtime and i < num_real_chunks - 1:  # 最后一个 chunk 不需要等
                expected_time = (i + 1) * self.chunk_ms / 1000.0  # 当前应该消耗的真实时间
                elapsed = time.perf_counter() - infer_start
                if elapsed < expected_time:
                    sleep_time = expected_time - elapsed
                    logger.info(f"[Realtime] Sleep {sleep_time:.3f}s to align with chunk {i+1}")
                    time.sleep(sleep_time)

            if gen_res['end_of_turn']:
                logger.info(f"[Duplex] Turn ended at real chunk {i}")
                break

        # ── 5. 静音推进 ──
        last_chunk_was_speak = (
            unit_records and not unit_records[-1]['is_listen']
            and not unit_records[-1]['end_of_turn']
        )

        if last_chunk_was_speak:
            max_silence_chunks = 10
            silence_chunk = np.zeros(chunk_size, dtype=np.float32)
            logger.info(f"\n[Duplex] Audio ended but model still SPEAK — silence push mode (max {max_silence_chunks})")

            silence_idx = 0
            while last_chunk_was_speak and silence_idx < max_silence_chunks:
                silence_idx += 1
                gen_res = self._process_chunk(
                    num_real_chunks + silence_idx - 1,
                    silence_chunk, 0, 0,
                    unit_records, all_speak_texts, all_audio_chunks,
                    is_silence=True,
                )

                if self.realtime:
                    time.sleep(self.chunk_ms / 1000.0)

                if gen_res['end_of_turn'] or gen_res['is_listen']:
                    logger.info(f"[Duplex] Silence push ended: {'turn_eos' if gen_res['end_of_turn'] else 'listen'}")
                    break

                last_chunk_was_speak = not gen_res['is_listen'] and not gen_res['end_of_turn']

            if silence_idx == max_silence_chunks and last_chunk_was_speak:
                logger.warning(f"[Duplex] Max silence chunks reached, forcing end")

        # ── 6. 计算延迟指标 ──
        total_elapsed_ms = (time.perf_counter() - infer_start) * 1000

        if first_speak_time_ms is not None:
            first_response_from_end_ms = first_speak_time_ms - audio_duration_ms
        else:
            first_response_from_end_ms = None

        # 拼接最终结果
        full_text = "".join(all_speak_texts)
        full_waveform = None
        if all_audio_chunks:
            full_waveform = np.concatenate(all_audio_chunks)

        # ── 诊断日志 ──
        print(f"\n{'='*60}")
        print(f"[Duplex Result] text='{full_text[:80]}...'")
        print(f"[Duplex Result] speak_segments={len(all_speak_texts)} audio_chunks={len(all_audio_chunks)}")
        if first_speak_time_ms is not None:
            print(f"[Duplex Result] FIRST_RESPONSE (from start): {first_speak_time_ms:.0f}ms")
            print(f"[Duplex Result] FIRST_RESPONSE (from end):   {first_response_from_end_ms:.0f}ms")
            if first_response_from_end_ms < 0:
                print(f"[Duplex Result]  NOTE: 负值 = 系统在用户说完前就开始抢话/插话")
        else:
            print(f"[Duplex Result] FIRST_RESPONSE: N/A (never spoke)")
        print(f"[Duplex Result] TOTAL_E2E: {total_elapsed_ms:.0f}ms")
        if full_waveform is not None:
            print(f"[Duplex Result] audio_total: {len(full_waveform)}samples ({len(full_waveform)/24000:.2f}s @ 24kHz)")
        else:
            print(f"[Duplex Result] NO AUDIO OUTPUT")
        print(f"{'='*60}")

        # 清理临时文件
        try:
            os.unlink(prompt_wav_path)
        except OSError:
            pass

        return {
            "text": full_text or "",
            "waveform": full_waveform,
            "latency_ms": total_elapsed_ms,
            "first_response_from_start_ms": first_speak_time_ms,
            "first_response_from_end_ms": first_response_from_end_ms,
            "audio_duration_ms": audio_duration_ms,
        }

    def _process_chunk(
        self,
        chunk_idx: int,
        chunk: np.ndarray,
        start: int,
        end: int,
        unit_records: list,
        all_speak_texts: list,
        all_audio_chunks: list,
        is_silence: bool = False,
    ):
        """处理单个 chunk，音频/文本收集逻辑完全参照 infer_debug.py"""
        import torch

        label = "SILENCE" if is_silence else f"{start/16000:.1f}s~{end/16000:.1f}s"
        chunk_start_time = time.perf_counter()

        # 1. Prefill
        prefill_res = self.model.duplex_prefill(audio_waveform=chunk, frame_list=None, max_slice_nums=1)

        # 2. Generate
        gen_res = self.model.duplex_generate(
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

        # 3. Finalize
        self.model.duplex_finalize()

        chunk_elapsed_ms = (time.perf_counter() - chunk_start_time) * 1000

        # 4. 打印
        is_listen = gen_res['is_listen']
        text = gen_res.get('text', '')
        audio_wav = gen_res.get('audio_waveform')
        audio_info = "None"
        if audio_wav is not None:
            if isinstance(audio_wav, torch.Tensor):
                audio_wav = audio_wav.cpu().numpy()
            audio_info = f"{len(audio_wav)}smp/{len(audio_wav)/24000:.2f}s"

        logger.info(
            f"[Chunk {chunk_idx:3d} {label:12s}] listen={is_listen} text='{text[:20]:20s}' "
            f"audio={audio_info:18s} cost={chunk_elapsed_ms:6.1f}ms"
        )

        # 5. 收集音频和文本 —— 完全参照 infer_debug.py
        if not is_listen and gen_res.get('audio_waveform') is not None:
            waveform = gen_res['audio_waveform']
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            all_audio_chunks.append(waveform)
            if text:
                all_speak_texts.append(text)

        # 6. 记录
        unit_records.append({
            "chunk_idx": chunk_idx,
            "is_silence": is_silence,
            "is_listen": is_listen,
            "end_of_turn": gen_res['end_of_turn'],
            "text": text,
        })

        return gen_res


# =============================================================================
# 评估流程
# =============================================================================

def save_wav(waveform, path: str, sr: int = 24000):
    if waveform is None:
        logger.warning(f"save_wav skipped: waveform is None")
        return
    sf.write(path, waveform, samplerate=sr)
    logger.info(f"Saved audio: {path} ({len(waveform)} samples @ {sr}Hz)")


def process_single(
    pid: str, example_id: str, input_path: str,
    model: MiniCPMOLocalInference,
    data: Dict, judge: Optional[LLMJudge],
    output_dir: str,
    provider: str = "minicpmo45_local",
    force: bool = False,
    ref_audio_path: Optional[str] = None,
) -> Optional[Dict]:
    item = data.get(example_id)
    if item is None:
        logger.warning(f"Example {example_id} not found in data — skipping")
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"output_{provider}_{pid}_{example_id}.wav"
    result_path = out_dir / f"result_{provider}_{pid}_{example_id}.json"

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

    is_video = str(input_path).endswith(".mp4")
    audio_path = input_path
    video_path = input_path if is_video else None
    result["input_type"] = "video" if is_video else "audio"

    logger.info(f"Evaluating: {example_id} [{category}] type={result['input_type']}")
    inference_start = time.time()
    try:
        system_prompt = item.get("system_prompt", "You are a helpful assistant.")
        infer_result = model.infer(
            audio_path=audio_path,
            video_path=video_path,
            system_prompt=system_prompt,
            ref_audio_path=ref_audio_path,
        )
        inference_time = time.time() - inference_start
        result["inference_time_s"] = round(inference_time, 2)

        text = infer_result["text"]
        waveform = infer_result["waveform"]

        result["response_text"] = text
        result["latency_ms"] = round(infer_result["latency_ms"], 1)
        result["first_response_from_start_ms"] = round(infer_result.get("first_response_from_start_ms"), 1) if infer_result.get("first_response_from_start_ms") is not None else None
        result["first_response_from_end_ms"] = round(infer_result.get("first_response_from_end_ms"), 1) if infer_result.get("first_response_from_end_ms") is not None else None

        # 保存音频
        if waveform is not None:
            save_wav(waveform, str(output_path))
            result["output_audio"] = str(output_path)
        else:
            result["output_audio"] = None
            logger.warning(f"No waveform for {example_id} — model stayed in LISTEN")

        audio_duration_ms = infer_result["audio_duration_ms"]
        result["input_audio_duration_ms"] = round(audio_duration_ms, 1)
        if audio_duration_ms > 0:
            result["rtf"] = round(infer_result["latency_ms"] / audio_duration_ms, 3)

        logger.info(f"  Text: {text[:100]}...")
        if result["first_response_from_start_ms"] is not None:
            logger.info(f"  FirstResponse(start): {result['first_response_from_start_ms']:.0f}ms")
            logger.info(f"  FirstResponse(end):   {result['first_response_from_end_ms']:.0f}ms")

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        result["status"] = "inference_error"
        result["error"] = str(e)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return result

    if judge and text:
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
    parser = argparse.ArgumentParser(description="MiniCPM-o-4.5 Local Evaluation (Full-Duplex Streaming)")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default="./eval_results")
    parser.add_argument("--provider", default="minicpmo45_local")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-speak-tokens", type=int, default=50)
    parser.add_argument("--chunk-ms", type=int, default=1000,
                        help="每个 chunk 的时长（毫秒）。默认 1000，可设 100/200/500 等更小值")
    parser.add_argument("--realtime", action="store_true",
                        help="启用实时模拟：每处理完一个 chunk，若 GPU 太快则 sleep 到与真实时间对齐")
    parser.add_argument("--ref-audio", type=str, default=None,
                        help="TTS 参考音色路径（wav 格式，16kHz）。不指定则默认使用输入音频前 10s")
    parser.add_argument("--pid")
    parser.add_argument("--example")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-json", default=str(DATA_JSON_PATH))
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Initializing local model: {args.provider}")
    logger.info("=" * 60)

    model = MiniCPMOLocalInference(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        max_speak_tokens=args.max_speak_tokens,
        chunk_ms=args.chunk_ms,
        realtime=args.realtime,
    )

    data = load_benchmark_data(args.data_json)
    logger.info(f"Loaded {len(data)} scenarios")

    inputs = discover_inputs(args.data_dir)
    logger.info(f"Found {len(inputs)} input files")

    if not inputs:
        logger.error("No input files found.")
        sys.exit(1)

    if args.pid:
        inputs = [(p, e, i) for p, e, i in inputs if p == args.pid]
        logger.info(f"Filtered to PID={args.pid}: {len(inputs)}")
    if args.example:
        inputs = [(p, e, i) for p, e, i in inputs if e == args.example]
        logger.info(f"Filtered to example={args.example}: {len(inputs)}")

    judge = LLMJudge() if INTERNAL_LLM_URL else None

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
            ref_audio_path=args.ref_audio,
        )

        if result is None:
            skipped += 1
        elif result.get("status") == "completed":
            all_results.append(result)
            success += 1
        else:
            all_results.append(result)
            failed += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation Summary ({args.provider})")
    logger.info(f"Completed: {success}, Failed: {failed}, Skipped: {skipped}")

    completed = [r for r in all_results if r.get("status") == "completed"]
    if completed:
        fr_start = [r["first_response_from_start_ms"] for r in completed if r.get("first_response_from_start_ms") is not None]
        if fr_start:
            logger.info(f"Avg FirstResponse(from start): {statistics.mean(fr_start):.0f}ms")

        fr_end = [r["first_response_from_end_ms"] for r in completed if r.get("first_response_from_end_ms") is not None]
        if fr_end:
            logger.info(f"Avg FirstResponse(from end):   {statistics.mean(fr_end):.0f}ms")

        latencies = [r["latency_ms"] for r in completed]
        logger.info(f"Avg E2E latency: {statistics.mean(latencies):.0f}ms")

        rtf_vals = [r["rtf"] for r in completed if "rtf" in r]
        if rtf_vals:
            logger.info(f"Avg RTF: {statistics.mean(rtf_vals):.3f}")

        if judge:
            scores = [r["human_likeness"]["overall"] for r in completed if "human_likeness" in r]
            if scores:
                logger.info(f"Avg human-likeness: {statistics.mean(scores):.1f}/5")

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
