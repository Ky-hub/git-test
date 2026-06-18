"""
scheduler.py
------------
全双工评测系统 · 调度模块

职责：运行在全双工交互循环中，输入模型回复 chunk，输出要发送给模型的 chunk。
只依赖 format.py 的数据类，不直接解析 JSON。

核心设计：
  - 状态机：SchedulerState 控制生命周期
  - 策略模式：每种 Transition 对应一个 TransitionHandler
  - 全双工并发：PLAYING_TURN 状态下，用户音频与模型回复允许同时存在
"""

from __future__ import annotations

import time
import enum
import queue
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any
from abc import ABC, abstractmethod

import sys
from pathlib import Path

# 将本文件所在目录加入 sys.path，确保能找到同级的 format.py
_FILE_DIR = Path(__file__).resolve().parent
if str(_FILE_DIR) not in sys.path:
    sys.path.insert(0, str(_FILE_DIR))

from format import (
    Chunk, ModelOutputChunk, Scenario, Turn, Transition,
    FixedTransition, ReactiveTransition, GlobalConfig,
    Content, Processing, Expected, TextExpectation,
    AudioExpectation, Action,
)


# ============================================================
# 1. 调度器状态机
# ============================================================

class SchedulerState(enum.Enum):
    IDLE = "idle"
    PLAYING_TURN = "playing_turn"                       # 正在发送当前 Turn 的音频
    WAITING_MODEL_RESPONSE = "waiting_model_response"   # 等待模型开始回复
    WAITING_MODEL_SILENCE = "waiting_model_silence"     # 模型回复中，等待静默
    TRANSITIONING = "transitioning"                       # 执行过渡（固定静音/环境音）
    INTERRUPTING = "interrupting"                         # 正在发送打断音频
    FINISHED = "finished"


# ============================================================
# 2. Transition 策略处理器（策略模式）
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
    """fixed + silence: 插入固定时长静音"""

    def __init__(self, duration_ms: int):
        self.duration_ms = duration_ms
        self._sent = False

    def should_advance(self, scheduler, model_chunk=None) -> bool:
        return self._sent

    def get_transition_audio(self, scheduler) -> List[Chunk]:
        self._sent = True
        return [Chunk.silence(self.duration_ms, source="fixed_silence")]


class FixedAmbientHandler(TransitionHandler):
    """fixed + ambient: 插入固定时长环境音"""

    def __init__(self, clip_id: str, duration_ms: int, volume_db: float = 0.0):
        self.clip_id = clip_id
        self.duration_ms = duration_ms
        self.volume_db = volume_db
        self._sent = False

    def should_advance(self, scheduler, model_chunk=None) -> bool:
        return self._sent

    def get_transition_audio(self, scheduler) -> List[Chunk]:
        self._sent = True
        return [
            Chunk.control(
                "request_ambient",
                {
                    "clip_id": self.clip_id,
                    "duration_ms": self.duration_ms,
                    "volume_db": self.volume_db,
                },
                source="fixed_ambient",
            )
        ]


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
# 3. 调度器核心
# ============================================================

class TurnScheduler:
    """
    全双工交互调度器。

    生命周期：
      1. load_scenario()    — 加载剧本并预注册音频
      2. on_model_chunk()   — 全双工循环中每收到模型 chunk 调用
      3. output_queue       — 异步消费要发送给模型的 chunk
    """

    def __init__(self):
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

    # ---------------------------------------------------------
    # 剧本加载
    # ---------------------------------------------------------
    def load_scenario(
        self,
        scenario: Scenario,
        audio_provider: Callable[[int, Turn], List[Chunk]],
    ) -> None:
        """
        加载剧本，并通过 audio_provider 回调预加载各 Turn 的音频 chunk。

        Args:
            scenario: 已校验的 Scenario 对象
            audio_provider: 接收 (turn_idx, turn) 返回 List[Chunk]
        """
        self.scenario = scenario
        self.turns = scenario.turns
        self.state = SchedulerState.IDLE
        self.current_turn_idx = -1

        # 预加载所有 Turn 音频
        for idx, turn in enumerate(self.turns):
            chunks = audio_provider(idx, turn)
            q = queue.Queue()
            for c in chunks:
                q.put(c)
            self._turn_audio_queues[idx] = q

        # 初始静音
        initial_silence = scenario.global_config.initial_silence_ms
        if initial_silence > 0:
            self.output_queue.put(Chunk.silence(initial_silence, source="initial_silence"))

    # ---------------------------------------------------------
    # 主入口：处理模型输入 chunk
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 状态处理子函数
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 辅助方法
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 公共 API
    # ---------------------------------------------------------
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
# 4. 全双工循环集成示例
# ============================================================

def run_full_duplex_loop_example() -> None:
    """
    展示如何在全双工交互循环中使用 TurnScheduler。
    """

    from format import ScenarioValidator

    # 1. 初始化
    scheduler = TurnScheduler()

    # 2. 音频提供回调（对接音频编辑子系统）
    def audio_provider(turn_idx: int, turn: Turn) -> List[Chunk]:
        clip_id = turn.content.base_clip_id
        return [Chunk.audio(b"fake_pcm", 0, 200, meta={"clip_id": clip_id})] * 10

    # 3. 构造并加载剧本
    raw_scenario = {
        "id": "test_001",
        "name": "唤醒后打断",
        "description": "测试唤醒与打断",
        "version": "1.0",
        "global": {"sample_rate": 24000, "initial_silence_ms": 1000},
        "turns": [
            {
                "turn_id": 0,
                "name": "唤醒",
                "description": "发送唤醒词",
                "content": {
                    "base_clip_id": "wake_word",
                    "text_ground_truth": "小爱同学",
                    "processing": {"fade_in_ms": 50, "fade_out_ms": 50},
                },
                "transition": {
                    "type": "reactive",
                    "reactive": {
                        "trigger": "after_model_speech",
                        "params": {"silence_threshold_ms": 600},
                    },
                },
                "expected": {
                    "response_type": "text",
                    "text": {
                        "should_contain": ["在", "嗯"],
                        "match_mode": "contains",
                    },
                },
            },
            {
                "turn_id": 1,
                "name": "打断",
                "description": "在模型回复中途打断",
                "content": {
                    "base_clip_id": "interrupt_stop",
                    "text_ground_truth": "停，换成陈奕迅的",
                    "processing": {"speed_ratio": 1.1, "fade_in_ms": 20},
                },
                "transition": {
                    "type": "reactive",
                    "reactive": {
                        "trigger": "fixed_delay_interrupt",
                        "params": {"inject_after_model_start_ms": 1500},
                    },
                },
                "expected": {
                    "response_type": "text",
                    "text": {
                        "should_contain": ["陈奕迅"],
                        "match_mode": "contains",
                    },
                    "actions": [{"type": "confirm"}],
                },
            },
        ],
    }

    scenario = ScenarioValidator.validate_json(raw_scenario)
    scheduler.load_scenario(scenario, audio_provider)

    # 4. 模拟模型输出流
    model_outputs = [
        ModelOutputChunk("state", None, state_tag="<speak>"),
        ModelOutputChunk("text", "在的，请说", duration_ms=300),
        ModelOutputChunk("audio", b"pcm", duration_ms=300, is_speech=True),
        ModelOutputChunk("audio", b"pcm", duration_ms=300, is_speech=False),
        ModelOutputChunk("state", None, state_tag="<listen>"),
        # Turn 1: 打断
        ModelOutputChunk("state", None, state_tag="<speak>"),
        ModelOutputChunk("text", "好的我正在", duration_ms=400),
    ]

    print("=" * 50)
    print("全双工交互循环开始")
    print("=" * 50)

    for i, model_chunk in enumerate(model_outputs):
        print(f"\n[循环 {i}] 模型输入: type={model_chunk.type}, "
              f"state={model_chunk.state_tag}, speech={model_chunk.is_speech}")

        to_send = scheduler.on_model_chunk(model_chunk)

        for chunk in to_send:
            print(f"  -> 调度输出: type={chunk.type}, duration={chunk.duration_ms}, "
                  f"meta={chunk.meta}")

        if scheduler.is_finished():
            print("\n[剧本结束]")
            break

    print("\n调度统计:", scheduler.get_stats())


if __name__ == "__main__":
    run_full_duplex_loop_example()
