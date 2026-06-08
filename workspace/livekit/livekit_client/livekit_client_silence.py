#!/usr/bin/env python3
"""
LiveKit Python 客户端 (配置驱动版 + 音频帧监听日志 + 静音检测)
重点：通过静音检测区分"有效音频"和"静音填充帧"，
只在真正收到声音时记录 first_frame / utterance，避免静音期日志爆炸。

【修复】utterance 检测基于"静音持续时间"而非"帧间间隔"，
解决静音帧持续更新 prev_frame_time 导致无法检测新 utterance 的问题。

依赖:
    pip install livekit livekit-api sounddevice numpy

配置文件 (config.json):
    {
        "livekit_url": "wss://...",
        "livekit_api_key": "...",
        "livekit_api_secret": "...",
        "room_name": "test-room",
        "agent_name": "python-agent",
        "log_dir": "./audio_logs",
        "log_flush_interval": 1.0,
        "utterance_gap_ms": 150,
        "silence_threshold": 100
    }
"""

import argparse
import asyncio
import json
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Set
from datetime import datetime

import numpy as np
import sounddevice as sd

from livekit import api, rtc
from livekit.rtc import (
    Room, RoomOptions, ConnectionState,
    Track, TrackPublication, TrackKind,
    AudioStream, AudioFrame, AudioFrameEvent,
    AudioSource, LocalAudioTrack,
)


# ========================== 配置 ==========================

class Config:
    def __init__(self, data: Dict[str, Any]):
        self.livekit_url = self._require(data, "livekit_url")
        self.livekit_api_key = self._require(data, "livekit_api_key")
        self.livekit_api_secret = self._require(data, "livekit_api_secret")
        self.room_name = self._require(data, "room_name")
        self.agent_name = self._require(data, "agent_name")
        self.sample_rate = data.get("sample_rate", 48000)
        self.channels = data.get("channels", 1)
        self.publish_mic = data.get("publish_mic", True)
        self.log_dir = data.get("log_dir", "./audio_logs")
        self.log_flush_interval = data.get("log_flush_interval", 1.0)
        self.utterance_gap_ms = data.get("utterance_gap_ms", 150)
        self.silence_threshold = data.get("silence_threshold", 100)

    @staticmethod
    def _require(data, key):
        if key not in data or not data[key]:
            raise ValueError(f"配置项缺失或为空: '{key}'")
        return str(data[key])

    @classmethod
    def from_json(cls, path):
        with open(Path(path), "r", encoding="utf-8") as f:
            return cls(json.load(f))


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


# ========================== 静音检测工具 ==========================

def is_silence_frame(frame: AudioFrame, threshold: int = 100) -> tuple[bool, int]:
    """
    检测音频帧是否为静音。
    返回: (是否静音, 峰值绝对值)
    """
    if not hasattr(frame.data, "__len__") or len(frame.data) == 0:
        return True, 0
    audio_data = np.asarray(frame.data, dtype=np.int16)
    if audio_data.size == 0:
        return True, 0
    peak = int(np.max(np.abs(audio_data)))
    return peak < threshold, peak


# ========================== 音频帧日志记录器 ==========================

class AudioFrameLogger:
    """
    事件类型：
        - session_start         : 音频流开始
        - first_frame           : 首个非静音帧（track 级别）
        - utterance_first_frame : 每次从静音恢复后的首个非静音帧
        - frame                 : 非静音普通帧
        - silence_period        : 连续静音期摘要
        - session_end           : 音频流结束
    """

    def __init__(self, log_dir: str, flush_interval: float = 1.0):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.flush_interval = flush_interval
        self._files: Dict[str, Any] = {}
        self._buffers: Dict[str, list] = {}
        self._lock = asyncio.Lock()
        self._session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._first_frame_logged: Set[str] = set()
        self._utterance_counter: Dict[str, int] = {}

    async def start(self):
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
        async with self._lock:
            for sid, buf in list(self._buffers.items()):
                if buf:
                    self._write_buffer(sid, buf)
                self._buffers[sid] = []
            for fh in self._files.values():
                try:
                    fh.flush()
                    fh.close()
                except Exception:
                    pass
            self._files.clear()

    def _get_filepath(self, track_sid: str, participant: str) -> Path:
        safe = participant.replace("/", "_").replace("\\", "_")
        return self.log_dir / f"{self._session_time}_{track_sid}_{safe}.jsonl"

    def _get_filehandle(self, track_sid: str, participant: str):
        if track_sid not in self._files:
            fp = self._get_filepath(track_sid, participant)
            self._files[track_sid] = open(fp, "w", encoding="utf-8", buffering=1)
            print(f"[LOG] 创建日志: {fp}")
        return self._files[track_sid]

    def _write_buffer(self, sid: str, buffer: list):
        if not buffer:
            return
        fh = self._files.get(sid)
        if fh:
            try:
                fh.write("\n".join(buffer) + "\n")
                fh.flush()
            except Exception as e:
                print(f"[!] 日志写入失败 ({sid}): {e}")

    async def _flush_loop(self):
        while self._running:
            await asyncio.sleep(self.flush_interval)
            async with self._lock:
                for sid, buf in list(self._buffers.items()):
                    if buf:
                        self._write_buffer(sid, buf)
                        self._buffers[sid] = []

    async def _append(self, track_sid: str, participant: str, record: dict, flush_now: bool = False):
        async with self._lock:
            if track_sid not in self._buffers:
                self._buffers[track_sid] = []
                self._get_filehandle(track_sid, participant)
            self._buffers[track_sid].append(json.dumps(record, ensure_ascii=False))
            if flush_now and self._buffers[track_sid]:
                self._write_buffer(track_sid, self._buffers[track_sid])
                self._buffers[track_sid] = []

    async def log_session_start(self, track_sid: str, participant: str,
                                sample_rate: int, num_channels: int):
        record = {
            "event": "session_start",
            "timestamp": time.time(),
            "timestamp_ns": time.time_ns(),
            "track_sid": track_sid,
            "participant_identity": participant,
            "sample_rate": sample_rate,
            "num_channels": num_channels,
            "silence_threshold": self._silence_threshold if hasattr(self, '_silence_threshold') else 100,
        }
        await self._append(track_sid, participant, record)

    async def log_first_frame(self, track_sid: str, participant: str, frame: AudioFrame, peak: int):
        if track_sid in self._first_frame_logged:
            return
        self._first_frame_logged.add(track_sid)
        self._utterance_counter[track_sid] = 1

        data_bytes = len(frame.data) if hasattr(frame.data, "__len__") else 0
        samples = getattr(frame, "samples_per_channel", data_bytes // (frame.num_channels * 2))
        duration_ms = (samples / frame.sample_rate) * 1000.0 if frame.sample_rate > 0 else 0.0
        now = time.time()

        record = {
            "event": "first_frame",
            "timestamp": now,
            "timestamp_ns": time.time_ns(),
            "track_sid": track_sid,
            "participant_identity": participant,
            "utterance_index": 1,
            "sample_rate": frame.sample_rate,
            "num_channels": frame.num_channels,
            "samples_per_channel": samples,
            "data_bytes": data_bytes,
            "duration_ms": round(duration_ms, 3),
            "peak_amplitude": peak,
            "iso_time": datetime.fromtimestamp(now).isoformat(),
        }
        await self._append(track_sid, participant, record, flush_now=True)

    async def log_utterance_first_frame(self, track_sid: str, participant: str,
                                        frame: AudioFrame, silence_duration_ms: float, peak: int):
        """【修复】基于静音持续时间检测新 utterance。"""
        self._utterance_counter[track_sid] = self._utterance_counter.get(track_sid, 0) + 1
        idx = self._utterance_counter[track_sid]

        data_bytes = len(frame.data) if hasattr(frame.data, "__len__") else 0
        samples = getattr(frame, "samples_per_channel", data_bytes // (frame.num_channels * 2))
        duration_ms = (samples / frame.sample_rate) * 1000.0 if frame.sample_rate > 0 else 0.0
        now = time.time()

        record = {
            "event": "utterance_first_frame",
            "timestamp": now,
            "timestamp_ns": time.time_ns(),
            "track_sid": track_sid,
            "participant_identity": participant,
            "utterance_index": idx,
            "silence_duration_ms": round(silence_duration_ms, 3),
            "sample_rate": frame.sample_rate,
            "num_channels": frame.num_channels,
            "samples_per_channel": samples,
            "data_bytes": data_bytes,
            "duration_ms": round(duration_ms, 3),
            "peak_amplitude": peak,
            "iso_time": datetime.fromtimestamp(now).isoformat(),
        }
        await self._append(track_sid, participant, record, flush_now=True)

    async def log_frame(self, track_sid: str, participant: str,
                        frame_index: int, frame: AudioFrame, peak: int):
        data_bytes = len(frame.data) if hasattr(frame.data, "__len__") else 0
        samples = getattr(frame, "samples_per_channel", data_bytes // (frame.num_channels * 2))
        duration_ms = (samples / frame.sample_rate) * 1000.0 if frame.sample_rate > 0 else 0.0

        record = {
            "event": "frame",
            "timestamp": time.time(),
            "timestamp_ns": time.time_ns(),
            "track_sid": track_sid,
            "participant_identity": participant,
            "frame_index": frame_index,
            "sample_rate": frame.sample_rate,
            "num_channels": frame.num_channels,
            "samples_per_channel": samples,
            "data_bytes": data_bytes,
            "duration_ms": round(duration_ms, 3),
            "peak_amplitude": peak,
        }
        await self._append(track_sid, participant, record)

    async def log_silence_period(self, track_sid: str, participant: str,
                                  silence_frames: int, duration_ms: float):
        record = {
            "event": "silence_period",
            "timestamp": time.time(),
            "timestamp_ns": time.time_ns(),
            "track_sid": track_sid,
            "participant_identity": participant,
            "silence_frames": silence_frames,
            "silence_duration_ms": round(duration_ms, 3),
        }
        await self._append(track_sid, participant, record)

    async def log_session_end(self, track_sid: str, participant: str,
                              total_frames: int, total_duration_ms: float):
        record = {
            "event": "session_end",
            "timestamp": time.time(),
            "timestamp_ns": time.time_ns(),
            "track_sid": track_sid,
            "participant_identity": participant,
            "total_utterances": self._utterance_counter.get(track_sid, 0),
            "total_frames": total_frames,
            "total_duration_ms": round(total_duration_ms, 3),
        }
        async with self._lock:
            fh = self._get_filehandle(track_sid, participant)
            if track_sid in self._buffers and self._buffers[track_sid]:
                self._write_buffer(track_sid, self._buffers[track_sid])
                self._buffers[track_sid] = []
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()


# ========================== 客户端核心 ==========================

class LiveKitClient:
    def __init__(self, config: Config):
        self.config = config
        self.room: Optional[Room] = None
        self._running = False
        self._tasks: list = []
        self.audio_streams: Dict[str, AudioStream] = {}
        self.audio_players: Dict[str, sd.OutputStream] = {}
        self.local_audio_source: Optional[AudioSource] = None
        self.local_audio_track: Optional[LocalAudioTrack] = None
        self.frame_logger = AudioFrameLogger(
            log_dir=config.log_dir,
            flush_interval=config.log_flush_interval,
        )

    async def connect(self):
        self.room = Room()
        self.room.on("connected", self._on_connected)
        self.room.on("disconnected", self._on_disconnected)
        self.room.on("connection_state_changed", self._on_connection_state_changed)
        self.room.on("participant_connected", self._on_participant_connected)
        self.room.on("participant_disconnected", self._on_participant_disconnected)
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_unsubscribed", self._on_track_unsubscribed)

        token = generate_token(self.config)
        print(f"[*] 正在连接房间: {self.config.room_name} @ {self.config.livekit_url}")
        await self.room.connect(self.config.livekit_url, token,
                                options=RoomOptions(auto_subscribe=True))
        print(f"[+] 已连接: {self.room.name}, identity={self.room.local_participant.identity}")
        self._running = True
        await self.frame_logger.start()

    async def publish_microphone(self):
        cfg = self.config
        print(f"[*] 发布麦克风 ({cfg.sample_rate}Hz, {cfg.channels}ch)...")
        self.local_audio_source = rtc.AudioSource(
            sample_rate=cfg.sample_rate, num_channels=cfg.channels)
        self.local_audio_track = LocalAudioTrack.create_audio_track(
            "microphone", self.local_audio_source)
        publish_options = rtc.TrackPublishOptions()
        publish_options.source = rtc.TrackSource.SOURCE_MICROPHONE
        publication = await self.room.local_participant.publish_track(
            self.local_audio_track, publish_options)
        print(f"[+] 麦克风已发布, sid={publication.sid}")
        task = asyncio.create_task(
            self._microphone_capture_task(cfg.sample_rate, cfg.channels), name="mic-capture")
        self._tasks.append(task)

    async def run(self):
        await self.connect()
        if self.config.publish_mic:
            try:
                await self.publish_microphone()
            except Exception as e:
                print(f"[!] 麦克风发布失败: {e}")
        print(f"\n[*] 运行中，silence_threshold={self.config.silence_threshold}, utterance_gap={self.config.utterance_gap_ms}ms")
        print("[*] 按 Ctrl+C 退出...")
        try:
            while self._running:
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    async def shutdown(self):
        if not self._running:
            return
        print("\n[*] 关闭中...")
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        for sid, player in list(self.audio_players.items()):
            try:
                player.stop()
                player.close()
            except Exception:
                pass
        self.audio_players.clear()
        self.audio_streams.clear()
        await self.frame_logger.stop()
        if self.room:
            await self.room.disconnect()
        print("[+] 已关闭")

    # ---------- 事件回调 ----------
    def _on_connected(self):
        print("[+] 房间连接成功")

    def _on_disconnected(self):
        print("[-] 房间断开")
        self._running = False

    def _on_connection_state_changed(self, state: ConnectionState):
        print(f"[*] 连接状态: {state}")

    def _on_participant_connected(self, participant):
        print(f"[+] 加入: {participant.identity}")

    def _on_participant_disconnected(self, participant):
        print(f"[-] 离开: {participant.identity}")

    def _on_track_subscribed(self, track: Track, publication: TrackPublication, participant):
        if track.kind != TrackKind.KIND_AUDIO:
            return
        print(f"[+] 订阅音频: {track.sid} from {participant.identity}")
        audio_stream = AudioStream(track)
        self.audio_streams[track.sid] = audio_stream
        asyncio.create_task(
            self._audio_playback_task(track.sid, audio_stream, participant.identity))

    def _on_track_unsubscribed(self, track: Track, publication: TrackPublication, participant):
        print(f"[-] 取消订阅: {track.sid}")
        if track.sid in self.audio_streams:
            del self.audio_streams[track.sid]
        if track.sid in self.audio_players:
            try:
                self.audio_players[track.sid].stop()
                self.audio_players[track.sid].close()
            except Exception:
                pass
            del self.audio_players[track.sid]

    # ---------- 音频任务 ----------
    async def _audio_playback_task(self, track_sid: str, audio_stream: AudioStream, identity: str):
        print(f"[*] 播放任务启动: {track_sid}")
        try:
            first_event = await audio_stream.__anext__()
        except StopAsyncIteration:
            print(f"[!] 空流: {track_sid}")
            return

        first_frame = first_event.frame
        sr = first_frame.sample_rate
        ch = first_frame.num_channels
        print(f"[+] 音频参数: {sr}Hz, {ch}ch, 首帧 {len(first_frame.data)} bytes")

        player = sd.OutputStream(samplerate=sr, channels=ch, dtype=np.int16, blocksize=0)
        player.start()
        self.audio_players[track_sid] = player

        await self.frame_logger.log_session_start(track_sid, identity, sr, ch)

        # ========== 【修复】状态机变量 ==========
        is_first_frame = True           # track 级别是否还没收到过非静音帧
        was_speaking = False            # 上一帧是否在说话（非静音）
        silence_streak = 0            # 连续静音帧计数
        silence_start_time = 0.0      # 本次静音期开始时间（绝对时间戳）
        frame_count = 0               # 有效（非静音）帧计数
        total_raw_frames = 0          # 原始帧计数（含静音）
        start_time = time.time()
        # ========================================

        async def _process_frame(frame: AudioFrame):
            nonlocal is_first_frame, was_speaking, silence_streak, silence_start_time
            nonlocal frame_count, total_raw_frames

            total_raw_frames += 1
            current_time = time.time()

            # ---- 静音检测 ----
            is_silence, peak = is_silence_frame(frame, self.config.silence_threshold)

            if is_silence:
                # ===== 静音帧 =====
                if was_speaking:
                    # 刚进入静音期
                    was_speaking = False
                    silence_streak = 1
                    silence_start_time = current_time
                else:
                    # 持续静音中
                    silence_streak += 1

                # 播放静音帧（保持播放器 buffer 不空）
                self._play_frame(player, frame)

                # 每 500 帧静音（约5秒@10ms/帧）记录一次摘要
                if silence_streak == 1 or silence_streak % 500 == 0:
                    silence_ms = (current_time - silence_start_time) * 1000.0 if silence_start_time > 0 else 0
                    await self.frame_logger.log_silence_period(
                        track_sid, identity, silence_streak, silence_ms)
                return

            # ===== 非静音帧（有效音频）=====
            if not was_speaking:
                # === 从静音期恢复 ===
                silence_duration_ms = 0.0
                if silence_start_time > 0:
                    silence_duration_ms = (current_time - silence_start_time) * 1000.0

                if is_first_frame:
                    # Track 级别首次有效音频
                    await self.frame_logger.log_first_frame(track_sid, identity, frame, peak)
                    is_first_frame = False
                    print(f"\n{'='*60}")
                    print(f"[FIRST FRAME] 首个有效音频帧!")
                    print(f"  Track SID : {track_sid}")
                    print(f"  From      : {identity}")
                    print(f"  时间戳(秒) : {current_time:.6f}")
                    print(f"  ISO 时间   : {datetime.fromtimestamp(current_time).isoformat()}")
                    print(f"  峰值幅度   : {peak} (threshold={self.config.silence_threshold})")
                    print(f"{'='*60}\n")
                elif silence_duration_ms > self.config.utterance_gap_ms:
                    # 【修复】基于静音持续时间判断新 utterance
                    await self.frame_logger.log_utterance_first_frame(
                        track_sid, identity, frame, silence_duration_ms, peak)
                    idx = self.frame_logger._utterance_counter.get(track_sid, 0)
                    print(f"\n{'='*60}")
                    print(f"[UTTERANCE #{idx}] 新段落开始!")
                    print(f"  静音期持续: {silence_duration_ms:.1f}ms (threshold={self.config.utterance_gap_ms}ms)")
                    print(f"  Track SID : {track_sid}")
                    print(f"  From      : {identity}")
                    print(f"  时间戳(秒) : {current_time:.6f}")
                    print(f"  ISO 时间   : {datetime.fromtimestamp(current_time).isoformat()}")
                    print(f"  峰值幅度   : {peak}")
                    print(f"{'='*60}\n")
                else:
                    # 静音太短，视为同一段话继续（不记录新 utterance）
                    pass

                was_speaking = True
                silence_streak = 0

            # 记录普通帧 + 播放
            frame_count += 1
            await self.frame_logger.log_frame(track_sid, identity, frame_count, frame, peak)
            self._play_frame(player, frame)

            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"[{track_sid}] 有效音频 {frame_count} 帧, 运行 {elapsed:.1f}s (原始帧 {total_raw_frames})")

        # 处理首帧
        await _process_frame(first_frame)

        try:
            async for event in audio_stream:
                if not self._running:
                    break
                await _process_frame(event.frame)
        except Exception as e:
            print(f"[!] 播放异常 ({track_sid}): {e}")
        finally:
            total_ms = (time.time() - start_time) * 1000.0
            print(f"[-] 播放结束: {track_sid}, 有效帧 {frame_count}, 原始帧 {total_raw_frames}, {total_ms:.1f}ms")
            await self.frame_logger.log_session_end(track_sid, identity, frame_count, total_ms)
            try:
                player.stop()
                player.close()
            except Exception:
                pass
            if track_sid in self.audio_players:
                del self.audio_players[track_sid]

    def _play_frame(self, player: sd.OutputStream, frame: AudioFrame):
        audio_data = np.asarray(frame.data, dtype=np.int16)
        if frame.num_channels > 1:
            audio_data = audio_data.reshape(-1, frame.num_channels)
        player.write(audio_data)

    async def _microphone_capture_task(self, sample_rate: int, channels: int):
        blocksize = sample_rate // 50
        bytes_per_sample = channels * 2
        overflow_count = 0
        try:
            with sd.RawInputStream(
                samplerate=sample_rate, channels=channels, dtype=np.int16,
                blocksize=blocksize, latency='low') as stream:
                print(f"[+] 麦克风启动 (blocksize={blocksize})")
                for _ in range(5):
                    stream.read(blocksize)
                while self._running:
                    data, overflowed = stream.read(blocksize)
                    if overflowed:
                        overflow_count += 1
                        if overflow_count <= 5:
                            print(f"[!] 溢出 x{overflow_count}")
                    else:
                        overflow_count = 0
                    audio_bytes = bytes(data)
                    frame = AudioFrame(
                        data=audio_bytes, sample_rate=sample_rate,
                        num_channels=channels,
                        samples_per_channel=len(audio_bytes) // bytes_per_sample)
                    await self.local_audio_source.capture_frame(frame)
                    await asyncio.sleep(0)
        except Exception as e:
            print(f"[!] 麦克风异常: {e}")
            import traceback
            traceback.print_exc()


# ========================== 入口 ==========================

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.json")
    args = parser.parse_args()
    try:
        config = Config.from_json(args.config)
    except Exception as e:
        print(f"[x] 配置错误: {e}")
        sys.exit(1)
    client = LiveKitClient(config)
    loop = asyncio.get_running_loop()

    def _signal_handler(sig):
        print(f"\n[!] 信号 {sig.name}")
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
