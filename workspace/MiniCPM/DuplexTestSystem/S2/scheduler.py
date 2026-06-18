"""
scheduler.py
------------
全双工评测系统 · 调度模块

职责：运行在全双工交互循环中，输入模型回复 chunk，输出要发送给模型的 chunk。
只依赖 format.py 的数据类，不直接解析 JSON。

所有配置从 config.json 读取。
"""

from __future__ import annotations

import sys
import struct
import json
import time
import enum
import queue
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any
from pathlib import Path
from abc import ABC, abstractmethod

# 确保能找到同级目录下的 format.py
_FILE_DIR = Path(__file__).resolve().parent
if str(_FILE_DIR) not in sys.path:
    sys.path.insert(0, str(_FILE_DIR))

from format import (
    Chunk,
    ModelOutputChunk,
    Scenario,
    Turn,
    Transition,
    FixedTransition,
    ReactiveTransition,
    GlobalConfig,
    Content,
    Processing,
    Expected,
    TextExpectation,
    AudioExpectation,
    Action,
    AudioLibrary,
    ScenarioValidator,
)


# ============================================================
# 1. 配置读取
# ============================================================

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    从 config.json 读取系统配置。
    
    期望结构：
    {
        "chunk_size_ms": 200,
        "audio_library_metadata": "./audio_library/metadata.json",
        "sample_rate": 24000
    }
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 默认值
    cfg.setdefault("chunk_size_ms", 200)
    cfg.setdefault("sample_rate", 24000)
    return cfg


# ============================================================
# 2. 调度器状态机
# ============================================================

class SchedulerState(enum.Enum):
    IDLE = "idle"
    PLAYING_TURN = "playing_turn"
    WAITING_MODEL_RESPONSE = "waiting_model_response"
    WAITING_MODEL_SILENCE = "waiting_model_silence"
    TRANSITIONING = "transitioning"
    INTERRUPTING = "interrupting"
    FINISHED = "finished"


# ============================================================
# 3. Transition 策略处理器（策略模式）
# ============================================================

class TransitionHandler(ABC):
    """
    每种 Transition 类型对应一个处理器。
    负责两件事：
      1. should_advance: 根据当前状态与模型 chunk，判断是否进入下一轮
      2. get_transition_audio: 获取过渡阶段需要发送的音频 chunk
    """

    @abstractmethod
    def should_advance(
        self,
        scheduler: TurnScheduler,
        model_chunk: Optional[ModelOutputChunk] = None,
    ) -> bool:
        pass

    @abstractmethod
    def get_transition_audio(self, scheduler: TurnScheduler) -> List[Chunk]:
        pass


class FixedSilenceHandler(TransitionHandler):
    """fixed + silence: 通过 AudioLibrary 生成固定大小的静音 chunk"""

    def __init__(self, duration_ms: int):
        self.duration_ms = duration_ms
        self._sent = False

    def should_advance(self, scheduler, model_chunk=None) -> bool:
        return self._sent

    def get_transition_audio(self, scheduler) -> List[Chunk]:
        self._sent = True
        return scheduler.audio_library.generate_silence(self.duration_ms)


class FixedAmbientHandler(TransitionHandler):
    """fixed + ambient: 通过 AudioLibrary 加载环境音并切成固定大小"""

    def __init__(self, clip_id: str, duration_ms: int, volume_db: float = 0.0):
        self.clip_id = clip_id
        self.duration_ms = duration_ms
        self.volume_db = volume_db
        self._sent = False

    def should_advance(self, scheduler, model_chunk=None) -> bool:
        return self._sent

    def get_transition_audio(self, scheduler) -> List[Chunk]:
        self._sent = True
        return scheduler.audio_library.load_ambient_audio(
            self.clip_id, self.duration_ms, self.volume_db
        )


class FixedSeamlessHandler(TransitionHandler):
    """fixed + seamless: 无过渡，立即衔接"""

    def should_advance(self, scheduler, model_chunk=None) -> bool:
        return True

    def get_transition_audio(self, scheduler) -> List[Chunk]:
        return []


class ReactiveAfterModelSpeechHandler(TransitionHandler):
    """
    reactive + after_model_speech
    基于输入的模型音频 chunk 做 VAD 静音累积检测。
    """

    def __init__(self, silence_threshold_ms: int):
        self.silence_threshold_ms = silence_threshold_ms
        self._silence_accumulated_ms = 0.0
        self._model_started = False

    def should_advance(self, scheduler, model_chunk=None) -> bool:
        if model_chunk is None:
            return False

        # 检测模型开始说话
        if model_chunk.type == "audio" and model_chunk.is_speech is True:
            self._model_started = True
            self._silence_accumulated_ms = 0.0
            return False

        if model_chunk.type == "state" and model_chunk.state_tag == "<speak>":
            self._model_started = True
            self._silence_accumulated_ms = 0.0
            return False

        # 累积静默
        if self._model_started:
            is_silence = (
                (model_chunk.type == "audio" and model_chunk.is_speech is False)
                or (model_chunk.type == "state" and model_chunk.state_tag == "<listen>")
            )
            if is_silence:
                self._silence_accumulated_ms += model_chunk.duration_ms
                if self._silence_accumulated_ms >= self.silence_threshold_ms:
                    return True
            elif model_chunk.type == "text":
                self._silence_accumulated_ms = 0.0

        return False

    def get_transition_audio(self, scheduler) -> List[Chunk]:
        return []


class ReactiveAfterModelSpeechPlusHandler(TransitionHandler):
    """
    reactive + after_model_speech_plus
    静默阈值满足后，再额外等待 post_silence_wait_ms。
    """

    def __init__(self, silence_threshold_ms: int, post_silence_wait_ms: int):
        self.base_handler = ReactiveAfterModelSpeechHandler(silence_threshold_ms)
        self.post_silence_wait_ms = post_silence_wait_ms
        self._phase = "waiting_silence"  # waiting_silence -> waiting_post -> ready
        self._post_wait_start = 0.0

    def should_advance(self, scheduler, model_chunk=None) -> bool:
        if self._phase == "waiting_silence":
            if self.base_handler.should_advance(scheduler, model_chunk):
                self._phase = "waiting_post"
                self._post_wait_start = time.monotonic() * 1000
            return False

        if self._phase == "waiting_post":
            elapsed = time.monotonic() * 1000 - self._post_wait_start
            return elapsed >= self.post_silence_wait_ms

        return True

    def get_transition_audio(self, scheduler) -> List[Chunk]:
        return []


class ReactiveFixedDelayInterruptHandler(TransitionHandler):
    """
    reactive + fixed_delay_interrupt
    从模型开始回复起，经过固定时间后强制打断。
    """

    def __init__(self, inject_after_model_start_ms: int):
        self.inject_after_model_start_ms = inject_after_model_start_ms
        self._model_start_time_ms: Optional[float] = None

    def should_advance(self, scheduler, model_chunk=None) -> bool:
        if model_chunk is None:
            return False

        # 检测模型开始回复
        if self._model_start_time_ms is None:
            is_start = (
                (model_chunk.type == "audio" and model_chunk.is_speech is True)
                or (model_chunk.type == "text" and len(str(model_chunk.payload)) > 0)
                or (model_chunk.type == "state" and model_chunk.state_tag == "<speak>")
            )
            if is_start:
                self._model_start_time_ms = time.monotonic() * 1000

        if self._model_start_time_ms is not None:
            elapsed = time.monotonic() * 1000 - self._model_start_time_ms
            return elapsed >= self.inject_after_model_start_ms

        return False

    def get_transition_audio(self, scheduler) -> List[Chunk]:
        return []


class ReactiveImmediateHandler(TransitionHandler):
    """reactive + immediate: 当前轮次播放完毕立即触发"""

    def should_advance(self, scheduler, model_chunk=None) -> bool:
        return True

    def get_transition_audio(self, scheduler) -> List[Chunk]:
        return []


# ============================================================
# 4. 音频库实现（基于 metadata.json）
# ============================================================

class MetadataAudioLibrary(AudioLibrary):
    """
    基于 metadata.json 索引的音频库实现。
    
    资源结构：
    audio_library/
        metadata.json      # 音频索引
        wake/
            wake_xiaoai.wav
        ambient/
            office_noise.wav
        ...
    """

    def __init__(
        self,
        metadata_path: str,
        chunk_size_ms: int = 200,
        sample_rate: int = 24000,
    ):
        self.metadata_path = Path(metadata_path)
        self.chunk_size_ms = chunk_size_ms
        self.sample_rate = sample_rate
        self.bytes_per_ms = sample_rate * 2 // 1000  # 16bit mono

        # 加载元数据索引
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self._meta = json.load(f)

        # 构建 clip_id -> clip_info 映射
        self._clip_map = {
            c["clip_id"]: c
            for c in self._meta.get("clips", [])
        }

    def _get_clip_path(self, clip_id: str) -> Path:
        """通过元数据索引查找文件绝对路径"""
        if clip_id not in self._clip_map:
            raise KeyError(
                f"音频库中未找到 clip_id: {clip_id}，"
                f"可用: {list(self._clip_map.keys())}"
            )

        info = self._clip_map[clip_id]
        file_path = info.get("file_path")

        # 如果 file_path 是绝对路径且存在，直接使用
        path = Path(file_path)
        if path.is_absolute() and path.exists():
            return path

        # 如果是相对路径，基于 metadata.json 所在目录解析
        base_dir = self.metadata_path.parent
        rel_path = info.get("relative_path", info.get("file_name"))
        alt_path = base_dir / rel_path
        if alt_path.exists():
            return alt_path

        raise FileNotFoundError(
            f"clip_id '{clip_id}' 的文件不存在: {file_path} 或 {alt_path}"
        )

    def _load_pcm(self, clip_id: str) -> bytes:
        """加载音频文件为 16bit mono PCM bytes"""
        wav_path = self._get_clip_path(clip_id)

        with open(wav_path, "rb") as f:
            data = f.read()

        # 解析 WAV 头获取实际 data offset
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            offset = 12
            while offset < len(data):
                chunk_id = data[offset : offset + 4]
                chunk_size = struct.unpack("<I", data[offset + 4 : offset + 8])[0]
                if chunk_id == b"data":
                    return data[offset + 8 : offset + 8 + chunk_size]
                offset += 8 + chunk_size
            raise ValueError(f"WAV 文件缺少 data chunk: {wav_path}")

        # 非 wav 格式，直接返回（假设原始 PCM）
        return data

    def _slice_pcm(self, pcm: bytes, clip_id: str) -> List[Chunk]:
        """将原始 PCM 按固定 chunk_size_ms 切割"""
        chunk_bytes = self.chunk_size_ms * self.bytes_per_ms
        chunks = []
        total = len(pcm)
        for i, offset in enumerate(range(0, total, chunk_bytes)):
            piece = pcm[offset : offset + chunk_bytes]
            actual_duration_ms = len(piece) // self.bytes_per_ms
            chunks.append(
                Chunk.audio(
                    pcm=piece,
                    timestamp_ms=i * self.chunk_size_ms,
                    duration_ms=actual_duration_ms,
                    meta={"clip_id": clip_id, "seq": i},
                )
            )
        return chunks

    def _apply_processing(self, chunks: List[Chunk], processing: Processing) -> List[Chunk]:
        """
        应用 Processing 配置。
        实际实现应接入音频处理流水线（变速、音量、淡入淡出等）。
        此处为占位，返回原样。
        """
        # TODO: 接入变速（speed_ratio）、音量（volume_db）、淡入淡出等
        return chunks

    def load_turn_audio(self, turn: Turn) -> List[Chunk]:
        """加载 Turn 音频：查索引 -> 加载 PCM -> 应用 Processing -> 切割"""
        pcm = self._load_pcm(turn.content.base_clip_id)
        chunks = self._slice_pcm(pcm, turn.content.base_clip_id)
        return self._apply_processing(chunks, turn.content.processing)

    def load_ambient_audio(
        self, clip_id: str, duration_ms: int, volume_db: float
    ) -> List[Chunk]:
        """加载环境音：查索引 -> 截取指定时长 -> 应用音量 -> 切割"""
        pcm = self._load_pcm(clip_id)

        # 按 duration_ms 截取（如果文件更长）
        max_bytes = duration_ms * self.bytes_per_ms
        pcm = pcm[:max_bytes]

        # TODO: 应用 volume_db 增益
        chunks = self._slice_pcm(pcm, clip_id)
        for c in chunks:
            c.meta["volume_db"] = volume_db
        return chunks

    def generate_silence(self, duration_ms: int) -> List[Chunk]:
        """生成空 PCM 静音，切成固定大小"""
        total_bytes = duration_ms * self.bytes_per_ms
        chunk_bytes = self.chunk_size_ms * self.bytes_per_ms
        chunks = []
        seq = 0
        for offset in range(0, total_bytes, chunk_bytes):
            piece_len = min(chunk_bytes, total_bytes - offset)
            actual_duration_ms = piece_len // self.bytes_per_ms
            # 16bit 空 PCM = 0x0000
            silence_pcm = b"\x00\x00" * (piece_len // 2)
            chunks.append(
                Chunk.audio(
                    pcm=silence_pcm,
                    timestamp_ms=seq * self.chunk_size_ms,
                    duration_ms=actual_duration_ms,
                    meta={"source": "generated_silence", "seq": seq},
                )
            )
            seq += 1
        return chunks

    def get_clip_info(self, clip_id: str) -> dict:
        """获取 clip 元数据（可用于校验 text_ground_truth 等）"""
        return self._clip_map.get(clip_id, {})


# ============================================================
# 5. 调度器核心
# ============================================================

class TurnScheduler:
    """
    全双工交互调度器。

    生命周期：
      1. load_scenario()    — 加载剧本并预注册音频
      2. on_model_chunk()   — 全双工循环中每收到模型 chunk 调用
      3. output_queue       — 异步消费要发送给模型的 chunk
    """

    def __init__(self, audio_library: AudioLibrary):
        """
        Args:
            audio_library: 音频资源库接口，负责所有音频资源的加载与固定大小切割。
        """
        self.audio_library = audio_library

        self.state = SchedulerState.IDLE
        self.current_turn_idx = -1
        self.scenario: Optional[Scenario] = None
        self.turns: List[Turn] = []

        # 音频缓存：turn_idx -> Queue[Chunk]
        self._turn_audio_queues: Dict[int, queue.Queue] = {}
        self._current_turn_audio_remaining: List[Chunk] = []

        # 输出缓冲
        self.output_queue: queue.Queue = queue.Queue()

        # 打断
        self._interrupt_audio_queue: queue.Queue = queue.Queue()
        self._is_interrupting = False

        # 统计
        self.stats = {
            "turn_start_times": {},
            "model_first_response_times": {},
            "interruptions": [],
        }

    def load_scenario(self, scenario: Scenario) -> None:
        """
        加载剧本，并通过 AudioLibrary 预加载所有音频资源。
        所有音频（Turn 内容、后续可能用到的环境音）在此阶段切成固定 chunk。
        """
        self.scenario = scenario
        self.turns = scenario.turns
        self.state = SchedulerState.IDLE
        self.current_turn_idx = -1

        # 通过 AudioLibrary 加载每个 Turn 的音频（已切成固定大小）
        for idx, turn in enumerate(self.turns):
            chunks = self.audio_library.load_turn_audio(turn)
            q = queue.Queue()
            for c in chunks:
                q.put(c)
            self._turn_audio_queues[idx] = q

        # 初始静音（同样走 AudioLibrary，切成固定 chunk）
        initial_silence = scenario.global_config.initial_silence_ms
        if initial_silence > 0:
            for c in self.audio_library.generate_silence(initial_silence):
                self.output_queue.put(c)

    def on_model_chunk(self, model_chunk: ModelOutputChunk) -> List[Chunk]:
        """
        全双工循环核心入口。

        返回: 本次调度决策产生的、需要立即发送给模型的 chunk 列表。
              这些 chunk 也会被追加到内部 output_queue。
        """
        output_chunks: List[Chunk] = []

        if self.state == SchedulerState.IDLE:
            output_chunks = self._handle_idle(model_chunk)
        elif self.state == SchedulerState.PLAYING_TURN:
            output_chunks = self._handle_playing_turn(model_chunk)
        elif self.state == SchedulerState.WAITING_MODEL_RESPONSE:
            output_chunks = self._handle_waiting_response(model_chunk)
        elif self.state == SchedulerState.WAITING_MODEL_SILENCE:
            output_chunks = self._handle_waiting_silence(model_chunk)
        elif self.state == SchedulerState.TRANSITIONING:
            output_chunks = self._handle_transitioning(model_chunk)
        elif self.state == SchedulerState.INTERRUPTING:
            output_chunks = self._handle_interrupting(model_chunk)
        elif self.state == SchedulerState.FINISHED:
            pass

        for c in output_chunks:
            self.output_queue.put(c)

        return output_chunks

    def _handle_idle(self, model_chunk: ModelOutputChunk) -> List[Chunk]:
        if self.current_turn_idx == -1:
            self._advance_to_turn(0)
        return []

    def _handle_playing_turn(self, model_chunk: ModelOutputChunk) -> List[Chunk]:
        chunks = []
        turn = self._get_current_turn()
        if not turn:
            return chunks

        # 1. 继续发送当前 Turn 音频（除非正在打断）
        if not self._is_interrupting:
            turn_chunk = self._consume_turn_audio()
            if turn_chunk:
                chunks.append(turn_chunk)

        # 2. 检查音频是否发完
        turn_audio_finished = self._is_turn_audio_finished()

        # 3. 处理 transition
        handler = self._get_transition_handler(turn.transition)

        if turn_audio_finished:
            if turn.transition.type == "fixed":
                # 固定过渡：发送过渡音频，然后进入下一轮
                trans_chunks = handler.get_transition_audio(self)
                chunks.extend(trans_chunks)
                if handler.should_advance(self, model_chunk):
                    self._advance_to_next_turn()
            elif turn.transition.type == "reactive":
                trigger = turn.transition.reactive.trigger if turn.transition.reactive else None
                if trigger == "immediate":
                    if handler.should_advance(self, model_chunk):
                        self._advance_to_next_turn()
                else:
                    # 进入等待模型状态
                    self.state = SchedulerState.WAITING_MODEL_RESPONSE
            else:
                self._advance_to_next_turn()

        # 4. 全双工并发：记录模型首响时间
        if model_chunk.type in ("audio", "text") and not turn_audio_finished:
            if self.current_turn_idx not in self.stats["model_first_response_times"]:
                self.stats["model_first_response_times"][self.current_turn_idx] = (
                    model_chunk.timestamp_ms
                )

        return chunks

    def _handle_waiting_response(self, model_chunk: ModelOutputChunk) -> List[Chunk]:
        is_model_active = (
            (model_chunk.type == "audio" and model_chunk.is_speech is True)
            or (model_chunk.type == "text" and len(str(model_chunk.payload)) > 0)
            or (model_chunk.type == "state" and model_chunk.state_tag == "<speak>")
        )

        if is_model_active:
            self.state = SchedulerState.WAITING_MODEL_SILENCE
            return self._handle_waiting_silence(model_chunk)

        return []

    def _handle_waiting_silence(self, model_chunk: ModelOutputChunk) -> List[Chunk]:
        turn = self._get_current_turn()
        if not turn:
            return []

        handler = self._get_transition_handler(turn.transition)

        if handler.should_advance(self, model_chunk):
            self._advance_to_next_turn()
            return self._start_next_turn_audio()

        return []

    def _handle_transitioning(self, model_chunk: ModelOutputChunk) -> List[Chunk]:
        return []

    def _handle_interrupting(self, model_chunk: ModelOutputChunk) -> List[Chunk]:
        chunks = []
        if not self._interrupt_audio_queue.empty():
            chunks.append(self._interrupt_audio_queue.get())
        else:
            self._is_interrupting = False
            self.state = SchedulerState.PLAYING_TURN
        return chunks

    def _get_current_turn(self) -> Optional[Turn]:
        if 0 <= self.current_turn_idx < len(self.turns):
            return self.turns[self.current_turn_idx]
        return None

    def _advance_to_turn(self, idx: int) -> None:
        if idx >= len(self.turns):
            self.state = SchedulerState.FINISHED
            self.output_queue.put(Chunk.control("scenario_finished"))
            return

        self.current_turn_idx = idx
        self.state = SchedulerState.PLAYING_TURN
        self.stats["turn_start_times"][idx] = time.monotonic() * 1000
        self._current_turn_audio_remaining = []

        q = self._turn_audio_queues.get(idx)
        if q:
            while not q.empty():
                self._current_turn_audio_remaining.append(q.get())

    def _advance_to_next_turn(self) -> None:
        self._advance_to_turn(self.current_turn_idx + 1)

    def _consume_turn_audio(self) -> Optional[Chunk]:
        if self._current_turn_audio_remaining:
            return self._current_turn_audio_remaining.pop(0)
        return None

    def _is_turn_audio_finished(self) -> bool:
        return len(self._current_turn_audio_remaining) == 0

    def _start_next_turn_audio(self) -> List[Chunk]:
        chunks = []
        for _ in range(2):
            c = self._consume_turn_audio()
            if c:
                chunks.append(c)
        return chunks

    def _get_transition_handler(self, transition: Transition) -> TransitionHandler:
        if transition.type == "fixed":
            ft = transition.fixed
            if ft is None:
                return FixedSeamlessHandler()
            if ft.mode == "silence":
                return FixedSilenceHandler(ft.silence_duration_ms or 0)
            if ft.mode == "ambient":
                return FixedAmbientHandler(
                    ft.ambient_clip_id or "",
                    ft.ambient_duration_ms or 0,
                    ft.ambient_volume_db,
                )
            return FixedSeamlessHandler()

        if transition.type == "reactive":
            rt = transition.reactive
            if rt is None:
                return FixedSeamlessHandler()

            trigger = rt.trigger
            params = rt.params

            if trigger == "after_model_speech":
                return ReactiveAfterModelSpeechHandler(params.get("silence_threshold_ms", 500))
            if trigger == "after_model_speech_plus":
                return ReactiveAfterModelSpeechPlusHandler(
                    params.get("silence_threshold_ms", 500),
                    params.get("post_silence_wait_ms", 0),
                )
            if trigger == "fixed_delay_interrupt":
                return ReactiveFixedDelayInterruptHandler(
                    params.get("inject_after_model_start_ms", 0)
                )
            if trigger == "immediate":
                return ReactiveImmediateHandler()

        return FixedSeamlessHandler()

    def is_finished(self) -> bool:
        return self.state == SchedulerState.FINISHED

    def get_current_turn_id(self) -> int:
        return self.current_turn_idx

    def get_stats(self) -> Dict[str, Any]:
        return self.stats

    def inject_interrupt_audio(self, chunks: List[Chunk]) -> None:
        """
        外部注入打断音频（如 fixed_delay_interrupt 场景下需要立即插入的音频）。
        由交互模拟子系统在检测到打断条件时调用。
        """
        for c in chunks:
            self._interrupt_audio_queue.put(c)
        self._is_interrupting = True
        self.state = SchedulerState.INTERRUPTING


# ============================================================
# 6. 全双工循环集成示例（从外部 JSON 读取剧本 + config）
# ============================================================

def load_scenario_from_json(path: str | Path) -> Scenario:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"剧本文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return ScenarioValidator.validate_json(raw)


def run_full_duplex_loop_example(
    scenario_path: str | Path,
    config_path: str | Path,
) -> None:
    """
    从 config.json 读取配置，从外部 JSON 读取剧本，运行全双工交互模拟。

    调用：
        python scheduler.py \
            --scenario scenario.json \
            --config config.json
    """
    # 1. 读取配置
    cfg = load_config(config_path)
    print(f"✅ 配置加载成功: chunk_size={cfg['chunk_size_ms']}ms")

    # 2. 加载剧本
    scenario = load_scenario_from_json(scenario_path)
    print(f"✅ 剧本加载成功: {scenario.id} ({scenario.name})")
    print(f"   共 {len(scenario.turns)} 轮")

    # 3. 初始化音频库（配置驱动）
    audio_lib = MetadataAudioLibrary(
        metadata_path=cfg["audio_library_metadata"],
        chunk_size_ms=cfg["chunk_size_ms"],
        sample_rate=cfg.get("sample_rate", 24000),
    )
    print(f"✅ 音频库加载成功: {len(audio_lib._clip_map)} 个 clip")

    # 4. 校验：剧本中的 base_clip_id 是否都在音频库中
    for turn in scenario.turns:
        clip_id = turn.content.base_clip_id
        if clip_id not in audio_lib._clip_map:
            print(f"⚠️ 警告: Turn {turn.turn_id} 的 clip_id '{clip_id}' 不在音频库中")
        else:
            info = audio_lib.get_clip_info(clip_id)
            if info.get("text_content") and info["text_content"] != turn.content.text_ground_truth:
                print(f"⚠️ 警告: Turn {turn.turn_id} text_ground_truth 与音频库不一致")

    # 5. 初始化调度器
    scheduler = TurnScheduler(audio_library=audio_lib)
    scheduler.load_scenario(scenario)

    # 6. 模拟模型输出流
    model_outputs = []
    for turn in scenario.turns:
        model_outputs.append(ModelOutputChunk("state", None, state_tag="<speak>"))
        model_outputs.append(ModelOutputChunk("text", f"回复第{turn.turn_id}轮", duration_ms=300))
        model_outputs.append(ModelOutputChunk("audio", b"pcm", duration_ms=300, is_speech=True))
        model_outputs.append(ModelOutputChunk("audio", b"pcm", duration_ms=300, is_speech=False))
        model_outputs.append(ModelOutputChunk("state", None, state_tag="<listen>"))

    print("=" * 50)
    print("全双工交互循环开始")
    print("=" * 50)

    for i, model_chunk in enumerate(model_outputs):
        print(
            f"\n[循环 {i}] 模型输入: type={model_chunk.type}, "
            f"state={model_chunk.state_tag}, speech={model_chunk.is_speech}"
        )

        to_send = scheduler.on_model_chunk(model_chunk)

        for chunk in to_send:
            print(
                f"  -> 调度输出: type={chunk.type}, duration={chunk.duration_ms}, "
                f"meta={chunk.meta}"
            )

        if scheduler.is_finished():
            print("\n[剧本结束]")
            break

    print("\n调度统计:", scheduler.get_stats())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全双工评测调度器")
    parser.add_argument("--scenario", "-s", required=True, help="剧本 JSON 文件路径")
    parser.add_argument("--config", "-c", required=True, help="配置文件 config.json 路径")
    args = parser.parse_args()

    run_full_duplex_loop_example(args.scenario, args.config)
