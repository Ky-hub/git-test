#!/usr/bin/env python3
"""
LiveKit 实时交互模拟评测脚本（零阻塞发送 + 接收音频保存版）
"""

import argparse
import asyncio
import json
import signal
import sys
import time
import wave
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum, auto

import numpy as np

from livekit import api, rtc
from livekit.rtc import (
    Room, RoomOptions, ConnectionState,
    Track, TrackPublication, TrackKind,
    AudioStream, AudioFrame, AudioSource, LocalAudioTrack,
)

# Windows 提升定时器精度到 1ms
if sys.platform == "win32":
    import ctypes
    ctypes.windll.winmm.timeBeginPeriod(1)


# ========================== 配置 ==========================

class Config:
    def __init__(self, data: Dict[str, Any]):
        self.livekit_url = self._require(data, "livekit_url")
        self.livekit_api_key = self._require(data, "livekit_api_key")
        self.livekit_api_secret = self._require(data, "livekit_api_secret")
        self.room_name = self._require(data, "room_name")
        self.agent_name = self._require(data, "agent_name")
        self.audio_file = self._require(data, "audio_file")
        self.chunk_duration_ms = data.get("chunk_duration_ms", 40)
        self.sample_rate = data.get("sample_rate", 24000)
        self.channels = data.get("channels", 1)
        self.silence_threshold = data.get("silence_threshold", 100)
        self.silence_confirmation_frames = data.get("silence_confirmation_frames", 30)
        self.log_dir = data.get("log_dir", "./eval_logs")
        self.save_recv_audio = data.get("save_recv_audio", True)
        self.wait_for_remote_timeout = data.get("wait_for_remote_timeout", 30.0)

    @staticmethod
    def _require(data, key):
        if key not in data or not data[key]:
            raise ValueError(f"配置项缺失或为空: '{key}'")
        return str(data[key])

    @classmethod
    def from_json(cls, path):
        with open(Path(path), "r", encoding="utf-8") as f:
            return cls(json.load(f))


# ========================== 时间格式化工具 ==========================

def fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "N/A"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# ========================== Token ==========================

def generate_token(config: Config) -> str:
    token = (
        api.AccessToken(config.livekit_api_key, config.livekit_api_secret)
        .with_identity(config.agent_name)
        .with_name(config.agent_name)
        .with_grants(
            api.VideoGrants(
                room_join=True, room=config.room_name,
                can_publish=True, can_subscribe=True,
            )
        )
    )
    return token.to_jwt()


# ========================== 静音检测 ==========================

def is_silence_frame(frame: AudioFrame, threshold: int = 100) -> tuple[bool, int]:
    if not hasattr(frame.data, "__len__") or len(frame.data) == 0:
        return True, 0
    audio_data = np.frombuffer(frame.data, dtype=np.int16)
    if audio_data.size == 0:
        return True, 0
    peak = int(np.max(np.abs(audio_data)))
    return peak < threshold, peak


# ========================== 日志记录器（零阻塞版）==========================

class EvalLogger:
    def __init__(self, log_dir: str, session_id: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self._lock = threading.Lock()
        self._buffer: list = []
        self._flush_interval = 0.5
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._file = None

    async def start(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._file = open(
            self.log_dir / f"eval_{self.session_id}_{ts}.jsonl",
            "w", encoding="utf-8", buffering=1
        )
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self):
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        with self._lock:
            self._flush()
            if self._file:
                self._file.close()
                self._file = None

    def _flush(self):
        if not self._buffer or not self._file:
            return
        try:
            self._file.write("\n".join(self._buffer) + "\n")
            self._file.flush()
            self._buffer.clear()
        except Exception as e:
            print(f"[!] 日志写入失败: {e}")

    async def _flush_loop(self):
        while self._running:
            await asyncio.sleep(self._flush_interval)
            with self._lock:
                self._flush()

    def log(self, record: dict):
        record["session_id"] = self.session_id
        if "timestamp" not in record:
            record["timestamp"] = time.time()
        if "timestamp_ns" not in record:
            record["timestamp_ns"] = time.time_ns()
        if "iso_time" not in record:
            record["iso_time"] = fmt_ts(record["timestamp"])
        self._buffer.append(json.dumps(record, ensure_ascii=False))

    def log_session_start(self, audio_file: str, chunk_duration_ms: int,
                          sample_rate: int, channels: int):
        self.log({
            "event": "session_start",
            "audio_file": audio_file,
            "chunk_duration_ms": chunk_duration_ms,
            "sample_rate": sample_rate,
            "num_channels": channels,
        })

    def log_chunk_sent(self, chunk_index: int, bytes_sent: int,
                       samples: int, is_last: bool,
                       scheduled_time_mono: float, actual_time_mono: float,
                       jitter_ms: float, interval_ms: float,
                       send_cost_ms: float, iso_time: str):
        self.log({
            "event": "chunk_sent",
            "chunk_index": chunk_index,
            "bytes_sent": bytes_sent,
            "samples": samples,
            "is_last": is_last,
            "scheduled_time_mono": round(scheduled_time_mono, 6),
            "actual_time_mono": round(actual_time_mono, 6),
            "jitter_ms": round(jitter_ms, 3),
            "interval_ms": round(interval_ms, 3),
            "send_cost_ms": round(send_cost_ms, 3),
            "iso_time": iso_time,
        })

    def log_recv_frame(self, track_sid: str, frame_index: int,
                       peak: int, is_silence: bool, is_first_valid: bool):
        self.log({
            "event": "recv_frame",
            "track_sid": track_sid,
            "frame_index": frame_index,
            "peak_amplitude": peak,
            "is_silence": is_silence,
            "is_first_valid": is_first_valid,
        })

    def log_first_valid_frame(self, track_sid: str, frame_index: int,
                              peak: int, silence_frames: int,
                              silence_duration_ms: float):
        self.log({
            "event": "first_valid_frame",
            "track_sid": track_sid,
            "frame_index": frame_index,
            "peak_amplitude": peak,
            "silence_frames_before": silence_frames,
            "silence_duration_ms": round(silence_duration_ms, 3),
        })

    def log_latency(self, first_response_latency_ms: float,
                    send_last_ts: float, recv_first_ts: float):
        self.log({
            "event": "latency_calculated",
            "first_response_latency_ms": round(first_response_latency_ms, 3),
            "send_last_timestamp": send_last_ts,
            "send_last_iso": fmt_ts(send_last_ts),
            "recv_first_timestamp": recv_first_ts,
            "recv_first_iso": fmt_ts(recv_first_ts),
        })

    def log_session_end(self, total_sent_chunks: int, total_recv_frames: int,
                        total_valid_frames: int,
                        jitter_stats: Optional[dict] = None):
        record = {
            "event": "session_end",
            "total_sent_chunks": total_sent_chunks,
            "total_recv_frames": total_recv_frames,
            "total_valid_frames": total_valid_frames,
        }
        if jitter_stats:
            record["jitter_stats"] = jitter_stats
        self.log(record)


# ========================== 状态机 ==========================

class RecvState(Enum):
    IDLE = auto()
    SILENCE_CANDIDATE = auto()
    SPEAKING = auto()


# ========================== 客户端核心 ==========================

class EvalClient:
    def __init__(self, config: Config):
        self.config = config
        self.room: Optional[Room] = None
        self._running = False
        self._tasks: list = []
        self.audio_source: Optional[AudioSource] = None
        self.audio_track: Optional[LocalAudioTrack] = None
        self.logger = EvalLogger(config.log_dir, config.agent_name)

        # 发送相关
        self._send_last_ts: Optional[float] = None
        self._total_sent_chunks = 0

        # 接收相关
        self._recv_first_ts: Optional[float] = None
        self._recv_total_frames = 0
        self._recv_valid_frames = 0
        self._latency_measured = False
        self._recv_state = RecvState.IDLE
        self._recv_silence_count = 0
        self._recv_silence_start = 0.0

        # 音频保存相关
        self._recv_audio_buffers: Dict[str, List[bytes]] = {}   # track_sid -> list of bytes
        self._recv_audio_params: Dict[str, dict] = {}           # track_sid -> {sr, ch, width}
        self._recv_audio_saved: set = set()

        # 同步原语
        self._recv_first_event = asyncio.Event()

    async def connect(self):
        self.room = Room()
        self.room.on("connected", self._on_connected)
        self.room.on("disconnected", self._on_disconnected)
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_unsubscribed", self._on_track_unsubscribed)

        token = generate_token(self.config)
        print(f"[*] [{now_iso()}] 连接房间: {self.config.room_name}")
        await self.room.connect(self.config.livekit_url, token,
                                options=RoomOptions(auto_subscribe=True))
        print(f"[+] [{now_iso()}] 已连接: identity={self.room.local_participant.identity}")
        self._running = True
        await self.logger.start()

    async def publish_audio_track(self):
        cfg = self.config
        print(f"[*] [{now_iso()}] 发布音频轨道 ({cfg.sample_rate}Hz, {cfg.channels}ch)...")
        self.audio_source = rtc.AudioSource(
            sample_rate=cfg.sample_rate, num_channels=cfg.channels)
        self.audio_track = LocalAudioTrack.create_audio_track(
            "simulated_audio", self.audio_source)
        publish_options = rtc.TrackPublishOptions()
        publish_options.source = rtc.TrackSource.SOURCE_MICROPHONE
        publication = await self.room.local_participant.publish_track(
            self.audio_track, publish_options)
        print(f"[+] [{now_iso()}] 音频轨道已发布, sid={publication.sid}")

    async def send_audio_file(self):
        """读取 WAV 文件并分 chunk 模拟实时发送（零阻塞版）"""
        cfg = self.config
        audio_path = Path(cfg.audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        with wave.open(str(audio_path), 'rb') as wf:
            file_sr = wf.getframerate()
            file_ch = wf.getnchannels()
            file_width = wf.getsampwidth()
            total_frames = wf.getnframes()
            raw_data = wf.readframes(total_frames)

        print(f"[+] [{now_iso()}] 加载音频: {audio_path}")
        print(f"    采样率={file_sr}Hz, 通道={file_ch}, 位深={file_width*8}bit, 总帧={total_frames}")

        if file_sr != cfg.sample_rate or file_ch != cfg.channels:
            print(f"[!] 警告: 文件参数({file_sr}Hz, {file_ch}ch)与配置({cfg.sample_rate}Hz, {cfg.channels}ch)不一致")

        bytes_per_sample = cfg.channels * 2
        samples_per_chunk = int(cfg.sample_rate * cfg.chunk_duration_ms / 1000)
        bytes_per_chunk = samples_per_chunk * bytes_per_sample

        audio_np = np.frombuffer(raw_data, dtype=np.int16)
        if file_ch != cfg.channels:
            if file_ch == 2 and cfg.channels == 1:
                audio_np = audio_np.reshape(-1, 2).mean(axis=1).astype(np.int16)
            else:
                raise ValueError("通道数转换不支持")

        if file_sr != cfg.sample_rate:
            ratio = cfg.sample_rate / file_sr
            new_len = int(len(audio_np) * ratio)
            audio_np = np.interp(
                np.linspace(0, len(audio_np), new_len, endpoint=False),
                np.arange(len(audio_np)),
                audio_np
            ).astype(np.int16)

        audio_bytes = audio_np.tobytes()
        total_len = len(audio_bytes)
        chunk_index = 0
        offset = 0

        self.logger.log_session_start(
            str(audio_path), cfg.chunk_duration_ms, cfg.sample_rate, cfg.channels
        )

        print(f"[*] [{now_iso()}] 开始发送音频，chunk_size={bytes_per_chunk} bytes, 预计 {total_len // bytes_per_chunk + 1} 个 chunk")

        start_mono = time.monotonic()
        interval_sec = cfg.chunk_duration_ms / 1000.0
        jitters: List[float] = []
        prev_actual_mono = start_mono
        overdue_count = 0
        warn_threshold_ms = cfg.chunk_duration_ms * 0.5

        while offset < total_len:
            if not self._running:
                print(f"[!] [{now_iso()}] 发送中断")
                break

            target_mono = start_mono + (chunk_index * interval_sec)

            end = min(offset + bytes_per_chunk, total_len)
            chunk = audio_bytes[offset:end]
            actual_samples = len(chunk) // bytes_per_sample
            frame = AudioFrame(
                data=chunk,
                sample_rate=cfg.sample_rate,
                num_channels=cfg.channels,
                samples_per_channel=actual_samples
            )

            now_mono = time.monotonic()
            sleep_needed = target_mono - now_mono

            if sleep_needed > 0.001:
                await asyncio.sleep(sleep_needed - 0.001)
                while time.monotonic() < target_mono:
                    pass
            elif sleep_needed > 0:
                while time.monotonic() < target_mono:
                    pass
            else:
                overdue_count += 1

            pre_send_mono = time.monotonic()
            pre_send_wall = time.time()
            await self.audio_source.capture_frame(frame)
            post_send_mono = time.monotonic()

            self._total_sent_chunks += 1
            is_last = (end >= total_len)

            jitter_ms = (pre_send_mono - target_mono) * 1000.0
            interval_ms = (pre_send_mono - prev_actual_mono) * 1000.0 if chunk_index > 0 else 0.0
            send_cost_ms = (post_send_mono - pre_send_mono) * 1000.0
            jitters.append(jitter_ms)
            prev_actual_mono = pre_send_mono

            pre_send_iso = fmt_ts(pre_send_wall)
            if chunk_index % 50 == 0 or abs(jitter_ms) > warn_threshold_ms or send_cost_ms > warn_threshold_ms:
                flag = ""
                if abs(jitter_ms) > warn_threshold_ms:
                    flag += " [JITTER]"
                if send_cost_ms > warn_threshold_ms:
                    flag += " [SLOW_SEND]"
                print(f"    [{pre_send_iso}] #{chunk_index:04d} | "
                      f"interval={interval_ms:6.2f}ms | jitter={jitter_ms:+6.2f}ms | "
                      f"send_cost={send_cost_ms:5.2f}ms{flag}")

            self.logger.log_chunk_sent(
                chunk_index=chunk_index,
                bytes_sent=len(chunk),
                samples=actual_samples,
                is_last=is_last,
                scheduled_time_mono=target_mono,
                actual_time_mono=pre_send_mono,
                jitter_ms=jitter_ms,
                interval_ms=interval_ms,
                send_cost_ms=send_cost_ms,
                iso_time=pre_send_iso,
            )

            if is_last:
                self._send_last_ts = pre_send_wall
                print(f"[+] [{pre_send_iso}] 最后一个 chunk 已发送 (#{self._total_sent_chunks})")
                self.logger.log({
                    "event": "send_complete",
                    "total_chunks": self._total_sent_chunks,
                    "send_last_timestamp": self._send_last_ts,
                    "send_last_iso": pre_send_iso,
                    "send_duration_ms": round((post_send_mono - start_mono) * 1000, 3),
                })

            chunk_index += 1
            offset = end

        if jitters:
            avg_jitter = sum(jitters) / len(jitters)
            max_jitter = max(jitters, key=abs)
            min_jitter = min(jitters, key=abs)
            std_jitter = (sum((j - avg_jitter) ** 2 for j in jitters) / len(jitters)) ** 0.5
            print(f"\n{'='*60}")
            print(f"[SEND STATS] 发送抖动统计")
            print(f"  总 chunk 数 : {len(jitters)}")
            print(f"  平均抖动    : {avg_jitter:+.3f} ms")
            print(f"  最小抖动    : {min_jitter:+.3f} ms")
            print(f"  最大抖动    : {max_jitter:+.3f} ms")
            print(f"  标准差      : {std_jitter:.3f} ms")
            print(f"  欠载次数    : {overdue_count} (错过目标时间)")
            print(f"{'='*60}\n")
            self._jitter_stats = {
                "avg_jitter_ms": round(avg_jitter, 3),
                "min_jitter_ms": round(min_jitter, 3),
                "max_jitter_ms": round(max_jitter, 3),
                "std_jitter_ms": round(std_jitter, 3),
                "overdue_count": overdue_count,
                "chunk_count": len(jitters),
            }
        else:
            self._jitter_stats = None

        print(f"[+] [{now_iso()}] 音频发送完成，共 {self._total_sent_chunks} 个 chunk")

    async def wait_for_first_response(self, timeout: float = 30.0):
        print(f"[*] [{now_iso()}] 等待首响 (timeout={timeout}s)...")
        try:
            await asyncio.wait_for(self._recv_first_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            print(f"[!] [{now_iso()}] 等待首响超时 ({timeout}s)")
            return False

    async def run(self):
        await self.connect()
        await self.publish_audio_track()

        recv_task = asyncio.create_task(self._recv_loop(), name="recv-loop")
        await asyncio.sleep(1.0)

        send_task = asyncio.create_task(self.send_audio_file(), name="send-audio")
        await send_task

        got_response = await self.wait_for_first_response(
            timeout=self.config.wait_for_remote_timeout
        )

        if got_response and self._send_last_ts and self._recv_first_ts:
            latency_ms = (self._recv_first_ts - self._send_last_ts) * 1000.0
            print(f"\n{'='*60}")
            print(f"[RESULT] 首响时延: {latency_ms:.3f} ms")
            print(f"         发送完成: {fmt_ts(self._send_last_ts)}")
            print(f"         首帧到达: {fmt_ts(self._recv_first_ts)}")
            print(f"{'='*60}\n")
            self.logger.log_latency(latency_ms, self._send_last_ts, self._recv_first_ts)
        else:
            print(f"[!] [{now_iso()}] 未能计算首响时延")

        print(f"[*] [{now_iso()}] 等待 5 秒收集后续音频...")
        await asyncio.sleep(5.0)
        await self.shutdown()

    async def shutdown(self):
        if not self._running:
            return
        print(f"\n[*] [{now_iso()}] 关闭中...")
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # 保存所有尚未保存的接收音频
        if self.config.save_recv_audio:
            for track_sid in list(self._recv_audio_buffers.keys()):
                if track_sid not in self._recv_audio_saved:
                    self._save_recv_audio(track_sid)

        jitter_stats = getattr(self, '_jitter_stats', None)
        self.logger.log_session_end(
            self._total_sent_chunks, self._recv_total_frames, self._recv_valid_frames,
            jitter_stats=jitter_stats
        )
        await self.logger.stop()

        if self.room:
            await self.room.disconnect()
        print(f"[+] [{now_iso()}] 已关闭")

    # ---------- 音频保存工具 ----------

    def _save_recv_audio(self, track_sid: str, participant_identity: str = "unknown"):
        """将接收到的音频缓冲区保存为 WAV 文件"""
        if track_sid in self._recv_audio_saved:
            return
        self._recv_audio_saved.add(track_sid)

        buffer = self._recv_audio_buffers.get(track_sid)
        params = self._recv_audio_params.get(track_sid)
        if not buffer or not params:
            return

        total_bytes = sum(len(b) for b in buffer)
        if total_bytes == 0:
            print(f"[!] [{now_iso()}] 音频缓冲区为空，跳过保存: {track_sid}")
            return

        # 合并所有音频数据
        audio_data = b"".join(buffer)

        # 文件名
        safe_identity = participant_identity.replace("/", "_").replace("\\", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recv_{ts}_{track_sid}_{safe_identity}.wav"
        filepath = Path(self.config.log_dir) / filename

        try:
            with wave.open(str(filepath), 'wb') as wf:
                wf.setnchannels(params["channels"])
                wf.setsampwidth(params["width"])
                wf.setframerate(params["sample_rate"])
                wf.writeframes(audio_data)
            print(f"[+] [{now_iso()}] 接收音频已保存: {filepath}")
            print(f"    大小: {total_bytes} bytes, 时长: ~{total_bytes / params['sample_rate'] / params['width'] / params['channels']:.2f}s")
        except Exception as e:
            print(f"[!] [{now_iso()}] 保存音频失败 ({track_sid}): {e}")

    # ---------- 事件回调 ----------

    def _on_connected(self):
        print(f"[+] [{now_iso()}] 房间连接成功")

    def _on_disconnected(self):
        print(f"[-] [{now_iso()}] 房间断开")
        self._running = False

    def _on_track_subscribed(self, track: Track, publication: TrackPublication, participant):
        if track.kind != TrackKind.KIND_AUDIO:
            return
        print(f"[+] [{now_iso()}] 订阅音频: {track.sid} from {participant.identity}")
        audio_stream = AudioStream(track)
        task = asyncio.create_task(
            self._audio_recv_task(track.sid, audio_stream, participant.identity),
            name=f"recv-{track.sid}"
        )
        self._tasks.append(task)

    def _on_track_unsubscribed(self, track: Track, publication: TrackPublication, participant):
        print(f"[-] [{now_iso()}] 取消订阅: {track.sid}")
        # 取消订阅时立即保存该 track 的音频
        if self.config.save_recv_audio and track.sid in self._recv_audio_buffers:
            self._save_recv_audio(track.sid, participant.identity)

    # ---------- 接收任务 ----------

    async def _recv_loop(self):
        while self._running:
            await asyncio.sleep(0.5)

    async def _audio_recv_task(self, track_sid: str, audio_stream: AudioStream, identity: str):
        print(f"[*] [{now_iso()}] 接收任务启动: {track_sid}")
        frame_index = 0
        state = RecvState.IDLE
        silence_count = 0
        silence_start = 0.0
        first_frame = True

        try:
            async for event in audio_stream:
                if not self._running:
                    break

                frame = event.frame
                self._recv_total_frames += 1
                frame_index += 1

                # 保存音频数据
                if self.config.save_recv_audio:
                    # 首帧时记录音频参数
                    if first_frame:
                        first_frame = False
                        self._recv_audio_params[track_sid] = {
                            "sample_rate": frame.sample_rate,
                            "channels": frame.num_channels,
                            "width": 2,  # int16 = 2 bytes
                        }
                        self._recv_audio_buffers[track_sid] = []
                        print(f"    [{now_iso()}] 开始缓存接收音频: {track_sid} "
                              f"({frame.sample_rate}Hz, {frame.num_channels}ch)")

                    # 将 frame data 转为 bytes 追加到缓冲区
                    if hasattr(frame.data, "tobytes"):
                        self._recv_audio_buffers[track_sid].append(frame.data.tobytes())
                    elif isinstance(frame.data, (bytes, bytearray)):
                        self._recv_audio_buffers[track_sid].append(bytes(frame.data))
                    else:
                        # 其他类型（如 memoryview），用 numpy 转
                        arr = np.asarray(frame.data, dtype=np.int16)
                        self._recv_audio_buffers[track_sid].append(arr.tobytes())

                is_silence, peak = is_silence_frame(frame, self.config.silence_threshold)

                if state == RecvState.IDLE:
                    if is_silence:
                        silence_count += 1
                        if silence_count == 1:
                            silence_start = time.time()
                    else:
                        state = RecvState.SPEAKING
                        self._recv_first_ts = time.time()
                        self._recv_valid_frames += 1
                        silence_duration = (self._recv_first_ts - silence_start) * 1000.0 if silence_count > 0 else 0.0
                        recv_iso = fmt_ts(self._recv_first_ts)

                        print(f"\n{'='*60}")
                        print(f"[FIRST VALID] 首响音频帧!")
                        print(f"  Track SID : {track_sid}")
                        print(f"  From      : {identity}")
                        print(f"  帧序号    : {frame_index}")
                        print(f"  峰值幅度  : {peak}")
                        print(f"  时间戳    : {recv_iso}")
                        print(f"  前置静音  : {silence_count} 帧 ({silence_duration:.1f}ms)")
                        print(f"{'='*60}\n")

                        self.logger.log_first_valid_frame(
                            track_sid, frame_index, peak, silence_count, silence_duration
                        )
                        self._recv_first_event.set()

                        if self._send_last_ts and not self._latency_measured:
                            self._latency_measured = True
                            latency_ms = (self._recv_first_ts - self._send_last_ts) * 1000.0
                            print(f"[LATENCY] 首响时延: {latency_ms:.3f} ms")
                            self.logger.log_latency(
                                latency_ms, self._send_last_ts, self._recv_first_ts
                            )

                        self.logger.log_recv_frame(
                            track_sid, frame_index, peak, False, True
                        )

                elif state == RecvState.SPEAKING:
                    if not is_silence:
                        self._recv_valid_frames += 1
                    self.logger.log_recv_frame(
                        track_sid, frame_index, peak, is_silence, False
                    )

                if frame_index % 100 == 0:
                    print(f"[{now_iso()}] [{track_sid}] 已接收 {frame_index} 帧, 有效 {self._recv_valid_frames} 帧")

        except Exception as e:
            print(f"[!] [{now_iso()}] 接收异常 ({track_sid}): {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"[-] [{now_iso()}] 接收结束: {track_sid}, 总帧 {frame_index}, 有效 {self._recv_valid_frames}")
            # 任务结束时保存音频（如果尚未保存）
            if self.config.save_recv_audio and track.sid in self._recv_audio_buffers:
                self._save_recv_audio(track_sid, identity)


# ========================== 入口 ==========================

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.json")
    args = parser.parse_args()

    try:
        config = Config.from_json(args.config)
    except Exception as e:
        print(f"[x] [{now_iso()}] 配置错误: {e}")
        sys.exit(1)

    client = EvalClient(config)
    loop = asyncio.get_running_loop()

    def _signal_handler(sig):
        print(f"\n[!] [{now_iso()}] 信号 {sig.name}")
        asyncio.create_task(client.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: _signal_handler(s))
        except NotImplementedError:
            pass

    if sys.platform == "win32":
        signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(client.shutdown()))
        signal.signal(signal.SIGTERM, lambda s, f: asyncio.create_task(client.shutdown()))

    try:
        await client.run()
    except KeyboardInterrupt:
        await client.shutdown()
    finally:
        await client.shutdown()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
