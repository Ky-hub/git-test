#!/usr/bin/env bash
"""MiniCPM-o 音色克隆批量测试 Gateway（固定语音 × 批量音色）

核心逻辑：
- user_audio_dir: 固定用户语音目录（剧本问题）
- voice_dir: 参考音色目录（自动扫描所有音频文件）
- 笛卡尔积：每个音色 × 每个用户语音
"""

import os
import re
import json
import asyncio
import argparse
import logging
import time
import base64
import zipfile
import glob
import hashlib
import wave
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager
from io import BytesIO

import yaml
import httpx
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gateway_batch_voice")


# ============================================================
# Worker 池
# ============================================================

class WorkerConnection:
    def __init__(self, address: str):
        self.address = address
        parts = address.split(":")
        self.host = parts[0]
        self.port = int(parts[1]) if len(parts) > 1 else 80
        self.url = f"http://{address}"
        self.status = "idle"
        self.current_job: Optional[str] = None
        self.total_requests = 0
        self.last_heartbeat = datetime.now()

    def mark_busy(self, job_id: str):
        self.status = "busy"
        self.current_job = job_id
        self.last_heartbeat = datetime.now()

    def mark_idle(self):
        self.status = "idle"
        self.current_job = None
        self.last_heartbeat = datetime.now()


class SimpleWorkerPool:
    def __init__(self, worker_addresses: List[str], request_timeout: float = 300.0):
        self.workers = [WorkerConnection(a) for a in worker_addresses]
        self.request_timeout = request_timeout
        self._lock = asyncio.Lock()

    async def acquire(self, job_id: str) -> WorkerConnection:
        while True:
            async with self._lock:
                for w in self.workers:
                    if w.status == "idle":
                        w.mark_busy(job_id)
                        return w
            await asyncio.sleep(0.2)

    def release(self, worker: WorkerConnection):
        worker.mark_idle()

    @property
    def idle_count(self) -> int:
        return sum(1 for w in self.workers if w.status == "idle")

    @property
    def busy_count(self) -> int:
        return sum(1 for w in self.workers if w.status == "busy")


# ============================================================
# 音频工具
# ============================================================

def load_audio_base64(path: str) -> str:
    import librosa
    audio, sr = librosa.load(path, sr=16000, mono=True)
    audio_bytes = audio.astype(np.float32).tobytes()
    return base64.b64encode(audio_bytes).decode("ascii")


def save_audio_base64(b64_data: str, dest_path: str, sample_rate: int = 16000):
    """保存 Worker 返回的音频数据

    Worker 可能返回两种格式：
    1. 完整的 WAV 文件（含 RIFF 头）→ 直接保存
    2. 裸 float32 PCM → 转为 int16 再加 WAV 头
    """
    raw_bytes = base64.b64decode(b64_data)

    # 检查是否已经是 WAV 文件
    if raw_bytes[:4] == b'RIFF' or raw_bytes[:4] == b'RIFX':
        logger.info(f"检测到 WAV 格式音频，直接保存 ({len(raw_bytes)} bytes)")
        with open(dest_path, "wb") as f:
            f.write(raw_bytes)
        return

    # 裸 float32 PCM 处理
    float_array = np.frombuffer(raw_bytes, dtype=np.float32)
    max_val = np.max(np.abs(float_array))

    if max_val < 1e-6:
        logger.warning(f"音频数据几乎全零，max={max_val}")
    elif max_val < 0.01:
        logger.warning(f"音频值域过小: {max_val}，可能是数据格式错误")
    elif max_val > 10:
        logger.warning(f"音频值域异常大: {max_val}，可能已经是 int16")
        # 尝试直接按 int16 保存
        if len(raw_bytes) % 2 == 0:
            int16_array = np.frombuffer(raw_bytes, dtype=np.int16)
            with wave.open(dest_path, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(int16_array.tobytes())
            return

    # 正常 float32 [-1, 1] → int16
    int16_array = (float_array * 32767).astype(np.int16)
    with wave.open(dest_path, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(int16_array.tobytes())
    logger.info(f"已保存 float32 PCM 音频: {dest_path}, 值域=[{-max_val:.4f}, {max_val:.4f}]")


def scan_audio_files(dir_path: str, recursive: bool = True) -> List[str]:
    """扫描目录下所有音频文件"""
    if not os.path.isdir(dir_path):
        raise ValueError(f"目录不存在: {dir_path}")
    exts = (".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac")
    if recursive:
        pattern = os.path.join(dir_path, "**", "*")
        all_files = glob.glob(pattern, recursive=True)
    else:
        all_files = glob.glob(os.path.join(dir_path, "*"))
    files = [f for f in all_files if os.path.isfile(f) and f.lower().endswith(exts)]
    files.sort()
    return files


# ============================================================
# 剧本配置管理
# ============================================================

class BatchConfigTemplate:
    def __init__(self, raw: Dict[str, Any]):
        self.id = raw.get("id", "")
        self.name = raw.get("name", "")
        self.user_audio_dir = raw.get("user_audio_dir", "")
        self.system_prompt = raw.get("system_prompt", "")
        self.voice_dir = raw.get("voice_dir", "")          # ← 音色目录
        self.voice_paths = raw.get("voice_paths", [])       # ← 额外指定单个文件
        self.output_dir = raw.get("output_dir", "")          # ← 输出目录
        self.voice_config = raw.get("voice_config", {})


def load_batch_configs(config_path: str) -> Dict[str, BatchConfigTemplate]:
    templates: Dict[str, BatchConfigTemplate] = {}
    if not os.path.isfile(config_path):
        return templates
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if raw and "batch_templates" in raw:
            for item in raw["batch_templates"]:
                tid = item.get("id")
                if tid:
                    templates[tid] = BatchConfigTemplate(item)
        logger.info(f"剧本配置加载完成: {len(templates)} 个模板")
    except Exception as e:
        logger.error(f"加载剧本配置失败: {e}")
    return templates


# ============================================================
# 批量任务模型
# ============================================================

class BatchJob:
    def __init__(self, job_id: str, user_audio_dir: str, output_dir: str,
                 system_prompt: str, voice_config: Dict[str, Any],
                 manifest_path: Optional[str] = None):
        self.id = job_id
        self.user_audio_dir = user_audio_dir
        self.output_dir = output_dir
        self.system_prompt = system_prompt
        self.voice_config = voice_config
        self.manifest_path = manifest_path  # 评测映射 JSON 路径
        self.status = "pending"
        self.tasks: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []
        self.progress = 0
        self.total = 0
        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None


# ============================================================
# 全局状态
# ============================================================

worker_pool: Optional[SimpleWorkerPool] = None
batch_templates: Dict[str, BatchConfigTemplate] = {}
jobs: Dict[str, BatchJob] = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(BASE_DIR, "data", "batch_jobs")
CONFIG_PATH = os.path.join(BASE_DIR, "config", "voice_batches.yaml")
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)


# ============================================================
# FastAPI
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global batch_templates
    batch_templates = load_batch_configs(CONFIG_PATH)
    logger.info(f"剧本模板: {len(batch_templates)} 个")
    yield
    logger.info("Gateway stopped")


app = FastAPI(
    title="MiniCPM-o Batch Voice Gateway",
    description="音色克隆批量测试（固定语音 × 批量音色）",
    version="2.2.0-batch-voice",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "idle_workers": worker_pool.idle_count if worker_pool else 0,
        "busy_workers": worker_pool.busy_count if worker_pool else 0,
    }


# ============================================================
# 剧本配置 API
# ============================================================

@app.get("/api/batch_configs")
async def list_batch_configs():
    return {
        "configs": [
            {
                "id": t.id,
                "name": t.name,
                "user_audio_dir": t.user_audio_dir,
                "voice_dir": t.voice_dir,
                "output_dir": t.output_dir,
                "system_prompt_preview": t.system_prompt[:60] + "..." if len(t.system_prompt) > 60 else t.system_prompt,
            }
            for t in batch_templates.values()
        ]
    }


@app.get("/api/batch_configs/{config_id}")
async def get_batch_config(config_id: str):
    t = batch_templates.get(config_id)
    if not t:
        raise HTTPException(status_code=404, detail="Config not found")
    return {
        "id": t.id, "name": t.name,
        "user_audio_dir": t.user_audio_dir,
        "voice_dir": t.voice_dir,
        "voice_paths": t.voice_paths,
        "output_dir": t.output_dir,
        "system_prompt": t.system_prompt,
        "voice_config": t.voice_config,
    }


@app.post("/api/batch_configs/reload")
async def reload_batch_configs():
    global batch_templates
    batch_templates = load_batch_configs(CONFIG_PATH)
    return {"success": True, "count": len(batch_templates)}


# ============================================================
# 扫描音色目录 API（前端用）
# ============================================================

@app.get("/api/scan_user_audios")
async def scan_user_audio_dir(user_audio_dir: str):
    """扫描服务器本地用户语音目录，返回所有音频文件（供前端多选）"""
    try:
        files = scan_audio_files(user_audio_dir)
        return {
            "dir": user_audio_dir,
            "count": len(files),
            "files": [
                {"path": f, "name": os.path.relpath(f, user_audio_dir)}
                for f in files
            ]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scan_voices")
async def scan_voice_dir(voice_dir: str):
    """扫描服务器本地音色目录，返回所有音频文件（供前端多选）"""
    try:
        files = scan_audio_files(voice_dir)
        return {
            "dir": voice_dir,
            "count": len(files),
            "voices": [
                {"path": f, "name": os.path.basename(f)}
                for f in files
            ]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/voice_preview")
async def preview_voice(path: str):
    """返回音色文件供前端试听（路径安全检查）"""
    real_path = os.path.realpath(path)
    if not os.path.isfile(real_path):
        raise HTTPException(status_code=404, detail="File not found")
    ext = os.path.splitext(real_path)[1].lower()
    mime = "audio/wav" if ext == ".wav" else "audio/mpeg" if ext == ".mp3" else "audio/ogg"
    return FileResponse(real_path, media_type=mime)


# ============================================================
# 批量任务 API（笛卡尔积）
# ============================================================

@app.post("/api/batch/jobs")
async def create_batch_job(
    background_tasks: BackgroundTasks,
    user_audio_dir: str = Form(..., description="服务器本地用户语音目录（固定问题）"),
    system_prompt: str = Form(default="你是一位乐于助人的助手。请用自然的中文语音回答用户的问题。"),
    voice_dir: Optional[str] = Form(None, description="音色目录（自动扫描所有音频）"),
    voice_paths: Optional[str] = Form(None, description="额外音色文件路径（逗号分隔，可选）"),
    selected_user_files: Optional[str] = Form(None, description="选中的用户语音文件（逗号分隔的相对路径，不传则跑全部）"),
    selected_voice_files: Optional[str] = Form(None, description="选中的音色文件（逗号分隔的相对/绝对路径，不传则跑全部）"),
    output_dir: Optional[str] = Form(None, description="输出目录（服务器本地绝对路径，默认自动生成）"),
    voice_config_json: Optional[str] = Form("{}"),
    recursive: bool = Form(True),
):
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        voice_config = json.loads(voice_config_json) if voice_config_json else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="voice_config_json 不是合法 JSON")

    # 收集音色路径
    all_voice_paths: List[str] = []

    # 1. 扫描音色目录
    if voice_dir:
        try:
            dir_voices = scan_audio_files(voice_dir, recursive=recursive)
            all_voice_paths.extend(dir_voices)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 2. 额外指定路径
    if voice_paths:
        extra = [v.strip() for v in voice_paths.split(",") if v.strip()]
        for p in extra:
            if os.path.exists(p):
                all_voice_paths.append(p)
            else:
                logger.warning(f"音色文件不存在，跳过: {p}")

    # 3. 如果前端传了 selected_voice_files，只保留选中的
    if selected_voice_files:
        selected_v_set = set(v.strip() for v in selected_voice_files.split(",") if v.strip())
        # 支持相对路径和绝对路径匹配
        filtered = []
        for p in all_voice_paths:
            rel = os.path.relpath(p, voice_dir) if voice_dir else p
            if p in selected_v_set or rel in selected_v_set or os.path.basename(p) in selected_v_set:
                filtered.append(p)
        all_voice_paths = filtered
        logger.info(f"音色过滤后: {len(all_voice_paths)} 个")

    # 去重
    seen = set()
    unique_voices = []
    for p in all_voice_paths:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            unique_voices.append(p)

    if not unique_voices:
        raise HTTPException(status_code=400, detail="未找到任何音色文件，请检查 voice_dir、voice_paths 或 selected_voice_files")

    # 扫描用户语音
    try:
        all_input_files = scan_audio_files(user_audio_dir, recursive=recursive)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 如果前端传了 selected_user_files，只保留选中的
    input_files = all_input_files
    if selected_user_files:
        selected_set = set(f.strip() for f in selected_user_files.split(",") if f.strip())
        filtered = []
        for p in all_input_files:
            rel = os.path.relpath(p, user_audio_dir)
            if p in selected_set or rel in selected_set or os.path.basename(p) in selected_set:
                filtered.append(p)
        input_files = filtered
        logger.info(f"用户语音过滤后: {len(input_files)} 个")

    if not input_files:
        raise HTTPException(status_code=400, detail=f"用户语音目录未找到音频文件: {user_audio_dir}")

    # 创建任务
    job_id = f"bv_{int(time.time()*1000)}"

    # 输出目录：用户指定 或 自动生成
    if output_dir:
        final_output_dir = output_dir
        manifest_path = os.path.join(final_output_dir, "manifest.json")
    else:
        final_output_dir = os.path.join(JOBS_DIR, job_id)
        manifest_path = os.path.join(final_output_dir, "manifest.json")

    os.makedirs(os.path.join(final_output_dir, "outputs"), exist_ok=True)

    job = BatchJob(
        job_id=job_id,
        user_audio_dir=user_audio_dir,
        output_dir=final_output_dir,
        system_prompt=system_prompt,
        voice_config=voice_config,
        manifest_path=manifest_path,
    )

    # 笛卡尔积：每个音色 × 每个用户语音
    for vpath in unique_voices:
        vname = os.path.basename(vpath)
        for fpath in input_files:
            fname = os.path.relpath(fpath, user_audio_dir)
            job.tasks.append({
                "voice_path": vpath,
                "voice_name": vname,
                "input_file": fpath,
                "input_name": fname,
            })

    job.total = len(job.tasks)
    jobs[job_id] = job

    background_tasks.add_task(run_batch_job, job)

    return {
        "job_id": job_id,
        "status": "pending",
        "total_tasks": job.total,
        "voices": len(unique_voices),
        "inputs": len(input_files),
        "system_prompt_preview": system_prompt[:60] + "..." if len(system_prompt) > 60 else system_prompt,
    }


async def run_batch_job(job: BatchJob):
    """后台执行：笛卡尔积（音色 × 用户语音），每次独立新 chat

    实时写入 manifest.json，方便评测时对照参考音色路径。
    """
    job.status = "running"
    out_dir = os.path.join(job.output_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    manifest_entries: List[Dict[str, Any]] = []

    for idx, task in enumerate(job.tasks):
        worker = await worker_pool.acquire(job.id)
        worker.total_requests += 1

        try:
            # 加载用户语音 + 参考音色
            audio_b64 = await asyncio.to_thread(load_audio_base64, task["input_file"])
            ref_audio_b64 = await asyncio.to_thread(load_audio_base64, task["voice_path"])

            payload = {
                "messages": [
                    {"role": "system", "content": job.system_prompt},
                    {"role": "user", "content": [{"type": "audio", "data": audio_b64}]},
                ],
                "generation": {
                    "max_new_tokens": job.voice_config.get("max_new_tokens", 256),
                    "do_sample": job.voice_config.get("do_sample", True),
                    "temperature": job.voice_config.get("temperature", 0.7),
                    "top_p": job.voice_config.get("top_p", 0.9),
                },
                "tts": {
                    "enabled": True,
                    "ref_audio_data": ref_audio_b64,
                },
            }

            async with httpx.AsyncClient(timeout=worker_pool.request_timeout) as client:
                resp = await client.post(f"{worker.url}/chat", json=payload)
                resp.raise_for_status()
                result = resp.json()

            out_text = result.get("text", "")
            out_audio_b64 = result.get("audio_data")
            success = result.get("success", False)

            # 输出命名：input_name + voice_name
            safe_input = re.sub(r'[^a-zA-Z0-9_.\-]', '_', os.path.splitext(task["input_name"])[0])
            safe_voice = re.sub(r'[^a-zA-Z0-9_.\-]', '_', os.path.splitext(task["voice_name"])[0])
            out_name = f"{safe_input}_{safe_voice}_out.wav"
            out_path = os.path.join(out_dir, out_name)

            if success and out_audio_b64:
                await asyncio.to_thread(save_audio_base64, out_audio_b64, out_path)

                # 写入 manifest 条目
                entry = {
                    "index": idx,
                    "input_audio": task["input_name"],           # 用户语音文件名
                    "input_path": task["input_file"],              # 用户语音完整路径
                    "ref_voice_name": task["voice_name"],          # 参考音色文件名
                    "ref_voice_path": task["voice_path"],          # 参考音色完整路径（评测关键）
                    "output_audio": out_name,                      # 输出音频文件名
                    "output_path": out_path,                       # 输出音频完整路径
                    "text": out_text,
                    "status": "success",
                }
                manifest_entries.append(entry)
                job.results.append({
                    "input": task["input_name"],
                    "voice": task["voice_name"],
                    "output": out_name,
                    "text": out_text,
                    "status": "success",
                })
            else:
                entry = {
                    "index": idx,
                    "input_audio": task["input_name"],
                    "input_path": task["input_file"],
                    "ref_voice_name": task["voice_name"],
                    "ref_voice_path": task["voice_path"],
                    "output_audio": None,
                    "output_path": None,
                    "text": out_text or "Worker 未返回音频",
                    "status": "no_audio" if success else "error",
                }
                manifest_entries.append(entry)
                job.results.append({
                    "input": task["input_name"],
                    "voice": task["voice_name"],
                    "output": None,
                    "text": out_text or "Worker 未返回音频",
                    "status": "no_audio" if success else "error",
                })

        except Exception as e:
            logger.error(f"[Batch {job.id}] 失败 {task['input_name']} + {task['voice_name']}: {e}", exc_info=True)
            entry = {
                "index": idx,
                "input_audio": task["input_name"],
                "input_path": task["input_file"],
                "ref_voice_name": task["voice_name"],
                "ref_voice_path": task["voice_path"],
                "output_audio": None,
                "output_path": None,
                "text": str(e),
                "status": "error",
            }
            manifest_entries.append(entry)
            job.results.append({
                "input": task["input_name"],
                "voice": task["voice_name"],
                "output": None,
                "text": str(e),
                "status": "error",
            })
        finally:
            worker_pool.release(worker)
            job.progress = idx + 1

            # 每完成 5 个或最后一个，刷新 manifest.json
            if (idx + 1) % 5 == 0 or idx == len(job.tasks) - 1:
                await asyncio.to_thread(_write_manifest, job, manifest_entries)

    job.status = "completed"
    job.completed_at = datetime.now()
    logger.info(f"[Batch {job.id}] 完成: {job.progress}/{job.total}")
    logger.info(f"[Batch {job.id}] 评测映射已保存: {job.manifest_path}")


def _write_manifest(job: BatchJob, entries: List[Dict[str, Any]]):
    """同步写入 manifest.json（评测映射文件）"""
    manifest = {
        "job_id": job.id,
        "user_audio_dir": job.user_audio_dir,
        "output_dir": job.output_dir,
        "system_prompt": job.system_prompt,
        "voice_config": job.voice_config,
        "total_tasks": job.total,
        "completed_tasks": len(entries),
        "created_at": job.created_at.isoformat(),
        "completed_at": datetime.now().isoformat() if job.status == "completed" else None,
        "mapping": entries,  # 核心：每个输出对应的参考音色路径
    }
    with open(job.manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


@app.get("/api/batch/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "total": job.total,
        "user_audio_dir": job.user_audio_dir,
        "output_dir": job.output_dir,
        "manifest_path": job.manifest_path,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "results": job.results,
    }


@app.get("/api/batch/jobs/{job_id}/manifest")
async def get_manifest(job_id: str):
    """获取评测映射文件 manifest.json"""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.manifest_path or not os.path.exists(job.manifest_path):
        raise HTTPException(status_code=404, detail="Manifest not found")
    with open(job.manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/batch/jobs/{job_id}/download")
async def download_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")

    out_dir = os.path.join(job.output_dir, "outputs")
    summary_path = os.path.join(job.output_dir, "results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "job_id": job.id,
            "system_prompt": job.system_prompt,
            "voice_config": job.voice_config,
            "results": job.results,
        }, f, ensure_ascii=False, indent=2)

    def build_zip():
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            if os.path.isdir(out_dir):
                for root, _dirs, files in os.walk(out_dir):
                    for fname in files:
                        abs_path = os.path.join(root, fname)
                        arc_name = os.path.join("outputs", os.path.relpath(abs_path, out_dir))
                        zf.write(abs_path, arc_name)
            # 评测映射文件（含参考音色路径）
            if job.manifest_path and os.path.exists(job.manifest_path):
                zf.write(job.manifest_path, "manifest.json")
            zf.write(summary_path, "results.json")
        buf.seek(0)
        return buf

    zip_buf = await asyncio.to_thread(build_zip)
    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={job_id}_results.zip"},
    )


@app.get("/api/batch/jobs/{job_id}/outputs/{filename}")
async def get_output_audio(job_id: str, filename: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    safe_name = os.path.basename(filename)
    file_path = os.path.join(job.output_dir, "outputs", safe_name)
    file_path = os.path.realpath(file_path)
    expected_root = os.path.realpath(os.path.join(job.output_dir, "outputs"))
    if not file_path.startswith(expected_root + os.sep) and file_path != expected_root:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/wav")


# ============================================================
# 静态文件
# ============================================================

static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    page_path = os.path.join(static_dir, "batch_voice_v2.html")
    if os.path.exists(page_path):
        return FileResponse(page_path)
    return HTMLResponse("<h1>Batch Voice Gateway</h1><p>请放置 static/batch_voice_v2.html</p>")


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MiniCPM-o Batch Voice Gateway")
    parser.add_argument("--port", type=int, default=10024, help="Gateway port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--workers", type=str, default="localhost:10031", help="Worker addresses, comma-separated")
    parser.add_argument("--timeout", type=float, default=300.0, help="Worker request timeout (s)")
    args = parser.parse_args()

    worker_list = [w.strip() for w in args.workers.split(",") if w.strip()]
    global worker_pool
    worker_pool = SimpleWorkerPool(worker_list, request_timeout=args.timeout)

    logger.info(f"Starting Batch Voice Gateway v2.2 on {args.host}:{args.port}")
    logger.info(f"Workers: {worker_list}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
