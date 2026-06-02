#!/usr/bin/env bash
"""MiniCPM-o 多模态剧本批量测试 Gateway（剧本=图片+音频 或 纯音频 × 单个参考音色）

核心逻辑：
- script_dir: 剧本根目录
  · 有子目录时：每个子目录是一个剧本（图片+音频）
  · 无子目录时：目录下每个音频文件是一个独立输入（纯音频）
- ref_voice_path: 单个参考音色文件
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
import wave
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager
from io import BytesIO

import yaml
import httpx
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gateway_batch_multimodal")


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
# 媒体工具
# ============================================================

def load_audio_base64(path: str) -> str:
    import librosa
    audio, sr = librosa.load(path, sr=16000, mono=True)
    audio_bytes = audio.astype(np.float32).tobytes()
    return base64.b64encode(audio_bytes).decode("ascii")


def load_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def save_audio_base64(b64_data: str, dest_path: str, sample_rate: int = 16000):
    raw_bytes = base64.b64decode(b64_data)
    if raw_bytes[:4] == b'RIFF' or raw_bytes[:4] == b'RIFX':
        with open(dest_path, "wb") as f:
            f.write(raw_bytes)
        return

    float_array = np.frombuffer(raw_bytes, dtype=np.float32)
    max_val = np.max(np.abs(float_array))

    if max_val > 10 and len(raw_bytes) % 2 == 0:
        int16_array = np.frombuffer(raw_bytes, dtype=np.int16)
        with wave.open(dest_path, 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sample_rate)
            wav.writeframes(int16_array.tobytes())
        return

    int16_array = (float_array * 32767).astype(np.int16)
    with wave.open(dest_path, 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sample_rate)
        wav.writeframes(int16_array.tobytes())


def scan_audio_files(dir_path: str, recursive: bool = False) -> List[str]:
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
# 剧本扫描（核心改造：支持纯音频扁平模式）
# ============================================================

def scan_scripts(script_dir: str) -> List[Dict[str, Any]]:
    """
    扫描剧本目录，自动识别两种结构：
    
    1. 子目录模式（多模态）：
       script_dir/
         ├── script_01/
         │     ├── pic1.jpg + audio.wav
         ├── script_02/
         │     ├── pic1.jpg + audio.wav
    
    2. 扁平模式（纯音频）：
       script_dir/
         ├── question1.wav
         ├── question2.mp3
         └── question3.wav
    
    返回统一格式：每个元素包含 name, path, images[], audio, image_count, mode
    """
    if not os.path.isdir(script_dir):
        raise ValueError(f"目录不存在: {script_dir}")

    img_exts = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
    audio_exts = (".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac")

    # 先检查是否有直接子目录
    subdirs = [os.path.join(script_dir, d) for d in sorted(os.listdir(script_dir))
               if os.path.isdir(os.path.join(script_dir, d))]

    scripts = []

    if subdirs:
        # ========== 子目录模式 ==========
        for subdir in subdirs:
            entry = os.path.basename(subdir)
            images = []
            audio = None

            for fname in sorted(os.listdir(subdir)):
                fpath = os.path.join(subdir, fname)
                if not os.path.isfile(fpath):
                    continue
                lower = fname.lower()
                if lower.endswith(img_exts):
                    images.append(fpath)
                elif lower.endswith(audio_exts) and audio is None:
                    audio = fpath

            scripts.append({
                "name": entry,
                "path": subdir,
                "images": images,
                "audio": audio,
                "image_count": len(images),
                "mode": "multimodal",
            })

    else:
        # ========== 扁平模式：目录下直接是音频文件 ==========
        audio_files = []
        for fname in sorted(os.listdir(script_dir)):
            fpath = os.path.join(script_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(audio_exts):
                audio_files.append(fpath)

        for fpath in audio_files:
            fname = os.path.basename(fpath)
            name = os.path.splitext(fname)[0]
            scripts.append({
                "name": name,
                "path": fpath,          # 这里 path 就是音频文件本身
                "images": [],
                "audio": fpath,
                "image_count": 0,
                "mode": "audio_only",
            })

    return scripts


# ============================================================
# 配置管理
# ============================================================

class BatchConfigTemplate:
    def __init__(self, raw: Dict[str, Any]):
        self.id = raw.get("id", "")
        self.name = raw.get("name", "")
        self.script_dir = raw.get("script_dir", "")
        self.ref_voice_path = raw.get("ref_voice_path", "")
        self.output_dir = raw.get("output_dir", "")
        self.system_prompt = raw.get("system_prompt", "")
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
    def __init__(self, job_id: str, script_dir: str, output_dir: str,
                 system_prompt: str, voice_config: Dict[str, Any],
                 ref_voice_path: str, manifest_path: Optional[str] = None):
        self.id = job_id
        self.script_dir = script_dir
        self.output_dir = output_dir
        self.system_prompt = system_prompt
        self.voice_config = voice_config
        self.ref_voice_path = ref_voice_path
        self.manifest_path = manifest_path
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
CONFIG_PATH = os.path.join(BASE_DIR, "config", "multimodal_batches.yaml")
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
    title="MiniCPM-o Batch Multimodal Gateway",
    description="多模态/纯音频剧本批量测试",
    version="3.1.0-batch-multimodal",
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


@app.get("/api/batch_configs")
async def list_batch_configs():
    return {
        "configs": [
            {
                "id": t.id,
                "name": t.name,
                "script_dir": t.script_dir,
                "ref_voice_path": t.ref_voice_path,
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
        "script_dir": t.script_dir,
        "ref_voice_path": t.ref_voice_path,
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
# 扫描 API（返回 mode 字段供前端识别）
# ============================================================

@app.get("/api/scan_scripts")
async def scan_script_dir(script_dir: str):
    try:
        scripts = scan_scripts(script_dir)
        return {
            "dir": script_dir,
            "count": len(scripts),
            "mode": "multimodal" if scripts and scripts[0]["mode"] == "multimodal" else "audio_only",
            "scripts": [
                {
                    "name": s["name"],
                    "path": s["path"],
                    "image_count": s["image_count"],
                    "images": [os.path.basename(i) for i in s["images"]],
                    "audio": os.path.basename(s["audio"]) if s["audio"] else None,
                    "mode": s["mode"],
                }
                for s in scripts
            ]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/voice_preview")
async def preview_voice(path: str):
    real_path = os.path.realpath(path)
    if not os.path.isfile(real_path):
        raise HTTPException(status_code=404, detail="File not found")
    ext = os.path.splitext(real_path)[1].lower()
    mime = "audio/wav" if ext == ".wav" else "audio/mpeg" if ext == ".mp3" else "audio/ogg"
    return FileResponse(real_path, media_type=mime)


# ============================================================
# 批量任务 API
# ============================================================

@app.post("/api/batch/jobs")
async def create_batch_job(
    background_tasks: BackgroundTasks,
    script_dir: str = Form(...),
    ref_voice_path: str = Form(...),
    system_prompt: str = Form(default="你是一位乐于助人的助手。请用自然的中文语音回答用户的问题。"),
    selected_scripts: Optional[str] = Form(None),
    output_dir: Optional[str] = Form(None),
    voice_config_json: Optional[str] = Form("{}"),
):
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    if not os.path.isfile(ref_voice_path):
        raise HTTPException(status_code=400, detail=f"参考音色文件不存在: {ref_voice_path}")

    try:
        voice_config = json.loads(voice_config_json) if voice_config_json else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="voice_config_json 不是合法 JSON")

    all_scripts = scan_scripts(script_dir)
    if not all_scripts:
        raise HTTPException(status_code=400, detail=f"目录下未找到任何剧本或音频: {script_dir}")

    scripts = all_scripts
    if selected_scripts:
        selected_set = set(s.strip() for s in selected_scripts.split(",") if s.strip())
        scripts = [s for s in all_scripts if s["name"] in selected_set]
        if not scripts:
            raise HTTPException(status_code=400, detail="未匹配到任何选中的剧本")

    # 过滤：至少要有音频
    valid_scripts = [s for s in scripts if s["audio"]]
    skipped = [s["name"] for s in scripts if not s["audio"]]
    if skipped:
        logger.warning(f"跳过无音频的条目: {skipped}")

    if not valid_scripts:
        raise HTTPException(status_code=400, detail="没有可用的输入（需至少包含1个音频）")

    job_id = f"bs_{int(time.time()*1000)}"
    if output_dir:
        final_output_dir = output_dir
    else:
        final_output_dir = os.path.join(JOBS_DIR, job_id)
    manifest_path = os.path.join(final_output_dir, "manifest.json")
    os.makedirs(os.path.join(final_output_dir, "outputs"), exist_ok=True)

    job = BatchJob(
        job_id=job_id,
        script_dir=script_dir,
        output_dir=final_output_dir,
        system_prompt=system_prompt,
        voice_config=voice_config,
        ref_voice_path=ref_voice_path,
        manifest_path=manifest_path,
    )

    for s in valid_scripts:
        job.tasks.append({
            "script_name": s["name"],
            "script_path": s["path"],
            "images": s["images"],
            "audio": s["audio"],
            "mode": s["mode"],
        })

    job.total = len(job.tasks)
    jobs[job_id] = job
    background_tasks.add_task(run_batch_job, job)

    return {
        "job_id": job_id,
        "status": "pending",
        "total_tasks": job.total,
        "scripts": len(valid_scripts),
        "ref_voice": os.path.basename(ref_voice_path),
        "system_prompt_preview": system_prompt[:60] + "..." if len(system_prompt) > 60 else system_prompt,
    }


async def run_batch_job(job: BatchJob):
    job.status = "running"
    out_dir = os.path.join(job.output_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    try:
        ref_audio_b64 = await asyncio.to_thread(load_audio_base64, job.ref_voice_path)
        logger.info(f"参考音色已加载: {job.ref_voice_path}")
    except Exception as e:
        logger.error(f"加载参考音色失败: {e}")
        job.status = "failed"
        return

    manifest_entries: List[Dict[str, Any]] = []

    for idx, task in enumerate(job.tasks):
        worker = await worker_pool.acquire(job.id)
        worker.total_requests += 1

        try:
            # 构造多模态 content：有图片就加图片，再叠加音频
            content = []
            if task["images"]:
                for img_path in task["images"]:
                    img_b64 = await asyncio.to_thread(load_image_base64, img_path)
                    content.append({"type": "image", "data": img_b64})

            audio_b64 = await asyncio.to_thread(load_audio_base64, task["audio"])
            content.append({"type": "audio", "data": audio_b64})

            payload = {
                "messages": [
                    {"role": "system", "content": job.system_prompt},
                    {"role": "user", "content": content},
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

            safe_name = re.sub(r'[^a-zA-Z0-9_.\-]', '_', task["script_name"])
            out_name = f"{safe_name}_out.wav"
            out_path = os.path.join(out_dir, out_name)

            if success and out_audio_b64:
                await asyncio.to_thread(save_audio_base64, out_audio_b64, out_path)

                entry = {
                    "index": idx,
                    "script_name": task["script_name"],
                    "script_path": task["script_path"],
                    "mode": task["mode"],
                    "images": [os.path.basename(i) for i in task["images"]],
                    "audio": os.path.basename(task["audio"]),
                    "output_audio": out_name,
                    "output_path": out_path,
                    "text": out_text,
                    "status": "success",
                }
                manifest_entries.append(entry)
                job.results.append({
                    "script": task["script_name"],
                    "images": len(task["images"]),
                    "audio": os.path.basename(task["audio"]),
                    "mode": task["mode"],
                    "output": out_name,
                    "text": out_text,
                    "status": "success",
                })
            else:
                entry = {
                    "index": idx,
                    "script_name": task["script_name"],
                    "script_path": task["script_path"],
                    "mode": task["mode"],
                    "images": [os.path.basename(i) for i in task["images"]],
                    "audio": os.path.basename(task["audio"]),
                    "output_audio": None,
                    "output_path": None,
                    "text": out_text or "Worker 未返回音频",
                    "status": "no_audio" if success else "error",
                }
                manifest_entries.append(entry)
                job.results.append({
                    "script": task["script_name"],
                    "images": len(task["images"]),
                    "audio": os.path.basename(task["audio"]),
                    "mode": task["mode"],
                    "output": None,
                    "text": out_text or "Worker 未返回音频",
                    "status": "no_audio" if success else "error",
                })

        except Exception as e:
            logger.error(f"[Batch {job.id}] 失败 {task['script_name']}: {e}", exc_info=True)
            entry = {
                "index": idx,
                "script_name": task["script_name"],
                "script_path": task["script_path"],
                "mode": task["mode"],
                "images": [os.path.basename(i) for i in task["images"]],
                "audio": os.path.basename(task["audio"]) if task["audio"] else None,
                "output_audio": None,
                "output_path": None,
                "text": str(e),
                "status": "error",
            }
            manifest_entries.append(entry)
            job.results.append({
                "script": task["script_name"],
                "images": len(task["images"]),
                "audio": os.path.basename(task["audio"]) if task["audio"] else None,
                "mode": task["mode"],
                "output": None,
                "text": str(e),
                "status": "error",
            })
        finally:
            worker_pool.release(worker)
            job.progress = idx + 1
            if (idx + 1) % 5 == 0 or idx == len(job.tasks) - 1:
                await asyncio.to_thread(_write_manifest, job, manifest_entries)

    job.status = "completed"
    job.completed_at = datetime.now()
    logger.info(f"[Batch {job.id}] 完成: {job.progress}/{job.total}")


def _write_manifest(job: BatchJob, entries: List[Dict[str, Any]]):
    manifest = {
        "job_id": job.id,
        "script_dir": job.script_dir,
        "ref_voice_path": job.ref_voice_path,
        "output_dir": job.output_dir,
        "system_prompt": job.system_prompt,
        "voice_config": job.voice_config,
        "total_tasks": job.total,
        "completed_tasks": len(entries),
        "created_at": job.created_at.isoformat(),
        "completed_at": datetime.now().isoformat() if job.status == "completed" else None,
        "mapping": entries,
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
        "script_dir": job.script_dir,
        "ref_voice_path": job.ref_voice_path,
        "output_dir": job.output_dir,
        "manifest_path": job.manifest_path,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "results": job.results,
    }


@app.get("/api/batch/jobs/{job_id}/manifest")
async def get_manifest(job_id: str):
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
    page_path = os.path.join(static_dir, "batch_multimodal.html")
    if os.path.exists(page_path):
        return FileResponse(page_path)
    return HTMLResponse("<h1>Batch Multimodal Gateway</h1><p>请放置 static/batch_multimodal.html</p>")


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MiniCPM-o Batch Multimodal Gateway")
    parser.add_argument("--port", type=int, default=10024)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--workers", type=str, default="localhost:10031")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    worker_list = [w.strip() for w in args.workers.split(",") if w.strip()]
    global worker_pool
    worker_pool = SimpleWorkerPool(worker_list, request_timeout=args.timeout)

    logger.info(f"Starting Batch Multimodal Gateway v3.1 on {args.host}:{args.port}")
    logger.info(f"Workers: {worker_list}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
