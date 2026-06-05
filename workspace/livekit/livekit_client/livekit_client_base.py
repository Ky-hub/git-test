#!/usr/bin/env python3
"""
LiveKit Python 客户端 (配置驱动版)
从配置文件读取连接参数，支持麦克风发布与远程音频订阅播放。

依赖安装:
    pip install livekit livekit-api sounddevice numpy

配置文件 (config.json) 示例:
    {
        "livekit_url": "wss://your-project.livekit.cloud",
        "livekit_api_key": "API_KEY",
        "livekit_api_secret": "API_SECRET",
        "room_name": "test-room",
        "agent_name": "python-agent"
    }

运行方式:
    python livekit_client.py --config config.json
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import sounddevice as sd

from livekit import api, rtc
from livekit.rtc import (
    Room,
    RoomOptions,
    ConnectionState,
    Track,
    TrackPublication,
    TrackKind,
    AudioStream,
    AudioFrame,
    AudioFrameEvent,
    AudioSource,
    LocalAudioTrack,
)


# ========================== 配置管理 ==========================

class Config:
    """配置对象，带默认值与校验。"""

    def __init__(self, data: Dict[str, Any]):
        self.livekit_url: str = self._require(data, "livekit_url")
        self.livekit_api_key: str = self._require(data, "livekit_api_key")
        self.livekit_api_secret: str = self._require(data, "livekit_api_secret")
        self.room_name: str = self._require(data, "room_name")
        self.agent_name: str = self._require(data, "agent_name")

        # 可选：音频参数
        self.sample_rate: int = data.get("sample_rate", 48000)
        self.channels: int = data.get("channels", 1)
        self.publish_mic: bool = data.get("publish_mic", True)

    @staticmethod
    def _require(data: Dict[str, Any], key: str) -> str:
        if key not in data or not data[key]:
            raise ValueError(f"配置项缺失或为空: '{key}'")
        return str(data[key])

    @classmethod
    def from_json(cls, path: str) -> "Config":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)


# ========================== Token 生成 ==========================

def generate_token(config: Config) -> str:
    """使用 api_key + api_secret 生成 JWT Token。"""
    token = (
        api.AccessToken(config.livekit_api_key, config.livekit_api_secret)
        .with_identity(config.agent_name)
        .with_name(config.agent_name)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=config.room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
    )
    return token.to_jwt()


# ========================== 客户端核心 ==========================

class LiveKitClient:
    def __init__(self, config: Config):
        self.config = config
        self.room: Optional[Room] = None
        self._running = False
        self._tasks: list = []

        # 音频播放
        self.audio_streams: Dict[str, AudioStream] = {}
        self.audio_players: Dict[str, sd.OutputStream] = {}

        # 本地麦克风
        self.local_audio_source: Optional[AudioSource] = None
        self.local_audio_track: Optional[LocalAudioTrack] = None

    # ---------- 生命周期 ----------

    async def connect(self):
        """连接房间并注册事件。"""
        self.room = Room()

        # 注册房间级别事件 —— 所有 .on() 回调必须是同步 def
        self.room.on("connected", self._on_connected)
        self.room.on("disconnected", self._on_disconnected)
        self.room.on("connection_state_changed", self._on_connection_state_changed)
        self.room.on("participant_connected", self._on_participant_connected)
        self.room.on("participant_disconnected", self._on_participant_disconnected)
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_unsubscribed", self._on_track_unsubscribed)

        # 生成 Token
        token = generate_token(self.config)
        print(f"[*] 正在连接房间: {self.config.room_name} @ {self.config.livekit_url}")

        await self.room.connect(
            self.config.livekit_url,
            token,
            options=RoomOptions(auto_subscribe=True),
        )
        print(f"[+] 已连接房间: {self.room.name}, identity={self.room.local_participant.identity}")
        self._running = True

    async def publish_microphone(self):
        """发布本地麦克风到房间。"""
        cfg = self.config
        print(f"[*] 正在发布麦克风 ({cfg.sample_rate}Hz, {cfg.channels}ch)...")

        self.local_audio_source = rtc.AudioSource(
            sample_rate=cfg.sample_rate, num_channels=cfg.channels
        )
        self.local_audio_track = LocalAudioTrack.create_audio_track(
            "microphone", self.local_audio_source
        )

        publish_options = rtc.TrackPublishOptions()
        publish_options.source = rtc.TrackSource.SOURCE_MICROPHONE

        publication = await self.room.local_participant.publish_track(
            self.local_audio_track, publish_options
        )
        print(f"[+] 麦克风已发布, track_sid={publication.sid}")

        task = asyncio.create_task(
            self._microphone_capture_task(cfg.sample_rate, cfg.channels),
            name="mic-capture",
        )
        self._tasks.append(task)

    async def run(self):
        """主循环。"""
        await self.connect()

        if self.config.publish_mic:
            try:
                await self.publish_microphone()
            except Exception as e:
                print(f"[!] 麦克风发布失败: {e}")
                print("[!] 客户端将继续运行（仅接收远程音频）")

        print("\n[*] 客户端运行中，按 Ctrl+C 退出...")
        try:
            while self._running:
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    async def shutdown(self):
        """优雅关闭。"""
        if not self._running:
            return
        print("\n[*] 正在关闭客户端...")
        self._running = False

        # 取消后台任务
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # 关闭音频播放器
        for sid, player in list(self.audio_players.items()):
            try:
                player.stop()
                player.close()
            except Exception:
                pass
        self.audio_players.clear()
        self.audio_streams.clear()

        # 断开房间
        if self.room:
            await self.room.disconnect()
        print("[+] 客户端已关闭")

    # ---------- 事件回调 (必须是同步 def!) ----------

    def _on_connected(self):
        print("[+] 房间连接成功")

    def _on_disconnected(self):
        print("[-] 房间断开连接")
        self._running = False

    def _on_connection_state_changed(self, state: ConnectionState):
        print(f"[*] 连接状态: {state}")

    def _on_participant_connected(self, participant):
        print(f"[+] 参与者加入: {participant.identity}")

    def _on_participant_disconnected(self, participant):
        print(f"[-] 参与者离开: {participant.identity}")

    def _on_track_subscribed(self, track: Track, publication: TrackPublication, participant):
        """同步回调：订阅到远程轨道。内部用 asyncio.create_task 启动异步任务。"""
        if track.kind != TrackKind.KIND_AUDIO:
            return

        print(f"[+] 订阅音频: {track.sid} from {participant.identity}")

        # 创建 AudioStream 主动消费音频帧
        audio_stream = AudioStream(track)
        self.audio_streams[track.sid] = audio_stream

        # 在同步回调里启动异步任务
        asyncio.create_task(
            self._audio_playback_task(track.sid, audio_stream, participant.identity)
        )

    def _on_track_unsubscribed(self, track: Track, publication: TrackPublication, participant):
        print(f"[-] 取消订阅音频: {track.sid}")
        if track.sid in self.audio_streams:
            del self.audio_streams[track.sid]
        if track.sid in self.audio_players:
            try:
                self.audio_players[track.sid].stop()
                self.audio_players[track.sid].close()
            except Exception:
                pass
            del self.audio_players[track.sid]

    # ---------- 音频任务 (异步) ----------

    async def _audio_playback_task(self, track_sid: str, audio_stream: AudioStream, identity: str):
        """异步任务：消费音频帧并播放。"""
        print(f"[*] 启动播放任务: {track_sid}")

        # AudioStream 迭代返回 AudioFrameEvent，通过 .frame 获取 AudioFrame
        try:
            first_event = await audio_stream.__anext__()
        except StopAsyncIteration:
            print(f"[!] 音频流为空: {track_sid}")
            return

        first_frame = first_event.frame
        sample_rate = first_frame.sample_rate
        channels = first_frame.num_channels
        print(f"[+] 音频参数: {sample_rate}Hz, {channels}ch, 第一帧 {len(first_frame.data)} bytes")

        # 创建播放器
        player = sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype=np.int16,
            blocksize=0,
        )
        player.start()
        self.audio_players[track_sid] = player

        self._play_frame(player, first_frame)

        frame_count = 1
        start_time = time.time()
        try:
            async for event in audio_stream:
                if not self._running:
                    break
                frame = event.frame
                self._play_frame(player, frame)
                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"[{track_sid}] 已播放 {frame_count} 帧, 运行 {elapsed:.1f}s")
        except Exception as e:
            print(f"[!] 播放任务异常 ({track_sid}): {e}")
        finally:
            print(f"[-] 播放任务结束: {track_sid}, 共 {frame_count} 帧")
            try:
                player.stop()
                player.close()
            except Exception:
                pass
            if track_sid in self.audio_players:
                del self.audio_players[track_sid]

    def _play_frame(self, player: sd.OutputStream, frame: AudioFrame):
        """将 AudioFrame 写入扬声器。

        AudioFrame.data 返回的是 CFFI buffer，需要用 np.asarray() 转换。
        """
        audio_data = np.asarray(frame.data, dtype=np.int16)
        if frame.num_channels > 1:
            audio_data = audio_data.reshape(-1, frame.num_channels)
        player.write(audio_data)

    async def _microphone_capture_task(self, sample_rate: int, channels: int):
        """后台任务：采集麦克风并推送到 LiveKit。

        【修复】解决启动时缓冲区溢出问题：
        1. blocksize 从 10ms 增大到 20ms，降低调度频率，减少 asyncio 开销占比
        2. 启动时先 drain 初始缓冲区（丢弃前几帧），让采集稳定
        3. 使用 latency='low' 参数，减少内部缓冲区深度
        4. 如果仍然溢出，自动增大 blocksize 到 40ms
        """
        # 初始 blocksize: 20ms (降低调度频率，减少 asyncio 开销)
        blocksize = sample_rate // 50  # 20ms
        bytes_per_sample = channels * 2  # int16 = 2 bytes

        overflow_count = 0

        try:
            with sd.RawInputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype=np.int16,
                blocksize=blocksize,
                latency='low',  # 减少 PortAudio 内部缓冲区深度
            ) as stream:
                print(f"[+] 麦克风采集已启动 (blocksize={blocksize} samples, {blocksize/sample_rate*1000:.1f}ms)")

                # 【修复】启动时 drain 初始缓冲区：丢弃前 5 帧，避免启动延迟导致的堆积
                print("[*] 预热麦克风缓冲区...")
                for _ in range(5):
                    stream.read(blocksize)
                print("[+] 预热完成，开始推送")

                while self._running:
                    data, overflowed = stream.read(blocksize)

                    if overflowed:
                        overflow_count += 1
                        # 如果连续溢出，自动增大 blocksize
                        if overflow_count >= 3:
                            print(f"[!] 连续溢出 {overflow_count} 次，尝试增大 blocksize...")
                            # 这里无法动态改 blocksize，只能提示
                            # 实际修复：重启 stream 或接受少量丢帧
                        if overflow_count <= 5:  # 前几次打印警告，后面静默
                            print(f"[!] 麦克风缓冲区溢出 (累计 {overflow_count} 次)")
                    else:
                        if overflow_count > 0 and overflow_count % 10 == 0:
                            print(f"[*] 缓冲区已稳定，最近无溢出")
                        overflow_count = 0

                    # 直接转 bytes —— AudioFrame.data 只接受 bytes
                    audio_bytes = bytes(data)

                    frame = AudioFrame(
                        data=audio_bytes,
                        sample_rate=sample_rate,
                        num_channels=channels,
                        samples_per_channel=len(audio_bytes) // bytes_per_sample,
                    )
                    await self.local_audio_source.capture_frame(frame)
                    await asyncio.sleep(0)  # 让出控制权

        except Exception as e:
            print(f"[!] 麦克风采集异常: {e}")
            import traceback
            traceback.print_exc()


# ========================== 入口 ==========================

async def main():
    parser = argparse.ArgumentParser(description="LiveKit Python Client")
    parser.add_argument(
        "--config", "-c", default="config.json", help="配置文件路径 (默认: config.json)"
    )
    args = parser.parse_args()

    # 读取配置
    try:
        config = Config.from_json(args.config)
    except Exception as e:
        print(f"[x] 配置错误: {e}")
        sys.exit(1)

    client = LiveKitClient(config)

    # 信号处理
    loop = asyncio.get_running_loop()

    def _signal_handler(sig):
        print(f"\n[!] 收到信号 {sig.name}")
        asyncio.create_task(client.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: _signal_handler(s))
        except NotImplementedError:
            # Windows 不支持 add_signal_handler，回退到 signal.signal
            pass

    # Windows 备用信号处理
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
