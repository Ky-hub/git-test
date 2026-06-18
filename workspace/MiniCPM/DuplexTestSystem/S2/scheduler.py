"""
scheduler.py
------------
全双工评测系统 · 调度模块

职责：运行在全双工交互循环中，输入模型回复 chunk，输出要发送给模型的 chunk。
所有音频资源（Turn 音频、环境音、静音）均通过 AudioLibrary 获取，且已切成固定大小。
"""

from __future__ import annotations

import sys
from pathlib import Path

_FILE_DIR = Path(__file__).resolve().parent
if str(_FILE_DIR) not in sys.path:
    sys.path.insert(0, str(_FILE_DIR))

import time
import enum
import queue
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any
from abc import ABC, abstractmethod

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
)


# ============================================================
# 1. 调度器状态机
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
# 2. Transition 策略处理器
# ============================================================

class TransitionHandler(ABC):
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
        # 通过音频库生成，返回的已是固定大小的 chunk 列表
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
    def should_advance(self, scheduler, model_chunk=None) -> bool:
        return True

    def get_transition_audio(self, scheduler) -> List[Chunk]:
        return []


class ReactiveAfterModelSpeechHandler(TransitionHandler):
    def __init__(self, silence_threshold_ms: int):
        self.silence_threshold_ms = silence_threshold_ms
        self._silence_accumulated_ms = 0.0
        self._model_started = False

    def should_advance(self, scheduler, model_chunk=None) -> bool:
        if model_chunk is None:
            return False

        if model_chunk.type == "audio" and model_chunk.is_speech is True:
            self._model_started = True
            self._silence_accumulated_ms = 0.0
            return False

        if model_chunk.type == "state" and model_chunk.state_tag == "<speak>":
            self._model_started = True
            self._silence_accumulated_ms = 0.0
            return False

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
    def __init__(self, silence_threshold_ms: int, post_silence_wait_ms: int):
        self.base_handler = ReactiveAfterModelSpeechHandler(silence_threshold_ms)
        self.post_silence_wait_ms = post_silence_wait_ms
        self._phase = "waiting_silence"
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
    def __init__(self, inject_after_model_start_ms: int):
        self.inject_after_model_start_ms = inject_after_model_start_ms
        self._model_start_time_ms: Optional[float] = None

    def should_advance(self, scheduler, model_chunk=None) -> bool:
        if model_chunk is None:
            return False

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
    def should_advance(self, scheduler, model_chunk=None) -> bool:
        return True

    def get_transition_audio(self, scheduler) -> List[Chunk]:
        return []


# ============================================================
# 3. 调度器核心
# ============================================================

class TurnScheduler:
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

        self.output_queue: queue.Queue = queue.Queue()

        self._interrupt_audio_queue: queue.Queue = queue.Queue()
        self._is_interrupting = False

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

        if not self._is_interrupting:
            turn_chunk = self._consume_turn_audio()
            if turn_chunk:
                chunks.append(turn_chunk)

        turn_audio_finished = self._is_turn_audio_finished()

        handler = self._get_transition_handler(turn.transition)

        if turn_audio_finished:
            if turn.transition.type == "fixed":
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
                    self.state = SchedulerState.WAITING_MODEL_RESPONSE
            else:
                self._advance_to_next_turn()

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
        for c in chunks:
            self._interrupt_audio_queue.put(c)
        self._is_interrupting = True
        self.state = SchedulerState.INTERRUPTING


# ============================================================
# 4. 示例：本地文件音频库实现 + 测试入口
# ============================================================

import json
import argparse
import struct


class LocalAudioLibrary(AudioLibrary):
    """
    AudioLibrary 的本地文件实现示例。
    
    资源目录结构（示例）：
    audio_library/
        wake_xiaoai.wav
        cmd_play_music.wav
        interrupt_stop.wav
        ambient_office.wav   # 环境音
    """

    def __init__(
        self,
        library_dir: str,
        chunk_size_ms: int = 200,
        sample_rate: int = 24000,
    ):
        self.library_dir = Path(library_dir)
        self.chunk_size_ms = chunk_size_ms
        self.sample_rate = sample_rate
        self.bytes_per_ms = sample_rate * 2 // 1000  # 16bit mono

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

    def _load_wav_pcm(self, clip_id: str) -> bytes:
        """简易 wav 读取，返回 16bit mono PCM bytes"""
        wav_path = self.library_dir / f"{clip_id}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"音频库缺少文件: {wav_path}")

        with open(wav_path, "rb") as f:
            data = f.read()

        # 简易解析：跳过 44 字节 WAV 头（生产环境应使用 wave 模块）
        return data[44:]

    def load_turn_audio(self, turn: Turn) -> List[Chunk]:
        pcm = self._load_wav_pcm(turn.content.base_clip_id)
        # 这里可接入 Processing（变速、音量、淡入淡出等）
        # 示例中直接原样切割
        return self._slice_pcm(pcm, turn.content.base_clip_id)

    def load_ambient_audio(
        self, clip_id: str, duration_ms: int, volume_db: float
    ) -> List[Chunk]:
        pcm = self._load_wav_pcm(clip_id)
        # 截取指定时长（如果文件更长）
        max_bytes = duration_ms * self.bytes_per_ms
        pcm = pcm[:max_bytes]
        # 这里可接入 volume_db 增益
        return self._slice_pcm(pcm, clip_id)

    def generate_silence(self, duration_ms: int) -> List[Chunk]:
        """生成空 PCM 静音，切成固定大小"""
        total_bytes = duration_ms * self.bytes_per_ms
        chunk_bytes = self.chunk_size_ms * self.bytes_per_ms
        chunks = []
        seq = 0
        for offset in range(0, total_bytes, chunk_bytes):
            piece_len = min(chunk_bytes, total_bytes - offset)
            actual_duration_ms = piece_len // self.bytes_per_ms
            # 空 PCM（16bit = 0x0000）
            silence_pcm = b"\x00\x00" * (piece_len // 2)
            chunks.append(
                Chunk.silence(
                    duration_ms=actual_duration_ms,
                    meta={"source": "generated_silence", "seq": seq},
                )
            )
            seq += 1
        return chunks


def load_scenario_from_json(path: str | Path) -> Scenario:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"剧本文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return ScenarioValidator.validate_json(raw)


def run_full_duplex_loop_example(scenario_path: str | Path, library_dir: str) -> None:
    """
    从外部 JSON 读取剧本，通过 LocalAudioLibrary 加载音频资源。
    
    调用：
        python scheduler.py --scenario scenario.json --library ./audio_library/
    """
    scenario = load_scenario_from_json(scenario_path)
    print(f"✅ 剧本加载成功: {scenario.id} ({scenario.name})")
    print(f"   共 {len(scenario.turns)} 轮")

    # 初始化音频库（固定 chunk 大小 200ms）
    audio_lib = LocalAudioLibrary(
        library_dir=library_dir,
        chunk_size_ms=200,
        sample_rate=scenario.global_config.sample_rate,
    )

    # 初始化调度器
    scheduler = TurnScheduler(audio_library=audio_lib)
    scheduler.load_scenario(scenario)

    # 模拟模型输出流
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
    parser.add_argument("--library", "-l", required=True, help="音频资源库目录路径")
    args = parser.parse_args()

    run_full_duplex_loop_example(args.scenario, args.library)
