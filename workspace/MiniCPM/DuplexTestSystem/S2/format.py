"""
format.py
---------
全双工评测系统 · 数据定义与剧本结构

职责：声明所有跨模块共享的数据结构，包括：
  1. 流式 Chunk（音频 / 文本 / 控制 / 静音）
  2. 模型输出 Chunk
  3. 剧本完整层级（Scenario → Turn → Content → Transition → Expected）
  4. 基础校验器

本文件不含任何调度、音频处理或 I/O 逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal, Union
from abc import ABC, abstractmethod


# ============================================================
# 1. 流式 Chunk（调度器输入输出的通用单元）
# ============================================================

@dataclass
class Chunk:
    """
    调度器与全双工循环之间的通用数据单元。
    音频编辑子系统产出、交互模拟子系统消费、调度器仲裁。
    """
    type: Literal["audio", "text", "control", "silence"]
    payload: Any = None
    timestamp_ms: float = 0.0          # 相对于本轮交互起始的时间戳
    duration_ms: float = 0.0           # 音频有效时长（仅 audio / silence）
    is_speech: Optional[bool] = None   # VAD 结果（仅 audio）

    # 追踪来源，用于日志与调试
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def audio(
        cls,
        pcm: bytes,
        timestamp_ms: float,
        duration_ms: float,
        is_speech: Optional[bool] = None,
        **meta: Any,
    ) -> Chunk:
        return cls(
            type="audio",
            payload=pcm,
            timestamp_ms=timestamp_ms,
            duration_ms=duration_ms,
            is_speech=is_speech,
            meta=meta,
        )

    @classmethod
    def text(cls, text: str, timestamp_ms: float, **meta: Any) -> Chunk:
        return cls(
            type="text",
            payload=text,
            timestamp_ms=timestamp_ms,
            duration_ms=0.0,
            meta=meta,
        )

    @classmethod
    def control(cls, signal: str, data: Any = None, **meta: Any) -> Chunk:
        return cls(
            type="control",
            payload={"signal": signal, "data": data},
            timestamp_ms=0.0,
            duration_ms=0.0,
            meta=meta,
        )

    @classmethod
    def silence(cls, duration_ms: float, **meta: Any) -> Chunk:
        return cls(
            type="silence",
            payload=None,
            timestamp_ms=0.0,
            duration_ms=duration_ms,
            meta=meta,
        )


@dataclass
class ModelOutputChunk:
    """
    模型回复的流式输出，由全双工循环输入给调度器。
    包含音频帧、文本 token、或状态占位符（如 <speak> / <listen>）。
    """
    type: Literal["audio", "text", "state"]
    payload: Any
    timestamp_ms: float = 0.0

    # 状态占位符（模型输出的特殊 token）
    state_tag: Optional[str] = None

    # 音频属性
    duration_ms: float = 0.0
    is_speech: Optional[bool] = None

    meta: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 2. 剧本层级结构（与说明书 1:1 映射）
# ============================================================

@dataclass
class BackgroundNoise:
    clip_id: str
    volume_db: float


@dataclass
class GlobalConfig:
    sample_rate: int = 24000
    initial_silence_ms: int = 1000
    background_noise: Optional[BackgroundNoise] = None


@dataclass
class Processing:
    """单段音频加工选项，声明式，由音频编辑子系统执行。"""
    denoise: bool = False
    normalize: bool = False
    speed_ratio: float = 1.0          # [0.5, 2.0]
    trim_silence: bool = False
    volume_db: float = 0.0            # [-20.0, 10.0]
    fade_in_ms: int = 0               # [0, 5000]
    fade_out_ms: int = 0              # [0, 5000]


@dataclass
class Content:
    base_clip_id: str
    text_ground_truth: str
    processing: Processing = field(default_factory=Processing)


# ------------------------------------------------------------
# Transition 联合体（固定 vs 反应式）
# ------------------------------------------------------------

@dataclass
class FixedTransition:
    mode: Literal["silence", "ambient", "seamless"]
    # mode == "silence"
    silence_duration_ms: Optional[int] = None
    # mode == "ambient"
    ambient_clip_id: Optional[str] = None
    ambient_duration_ms: Optional[int] = None
    ambient_volume_db: float = 0.0


@dataclass
class ReactiveTransition:
    trigger: Literal[
        "after_model_speech",
        "after_model_speech_plus",
        "fixed_delay_interrupt",
        "immediate",
    ]
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    type: Literal["fixed", "reactive"]
    fixed: Optional[FixedTransition] = None
    reactive: Optional[ReactiveTransition] = None


# ------------------------------------------------------------
# Expected 期望行为
# ------------------------------------------------------------

@dataclass
class TextExpectation:
    should_contain: List[str] = field(default_factory=list)
    should_not_contain: List[str] = field(default_factory=list)
    match_mode: Literal["contains", "exact", "regex"] = "contains"


@dataclass
class AudioExpectation:
    should_play: bool = False
    min_duration_ms: Optional[int] = None
    max_duration_ms: Optional[int] = None


@dataclass
class Action:
    type: Literal["confirm", "execute", "ask_back", "refuse"]


@dataclass
class Expected:
    response_type: Literal["text", "audio", "both", "none"]
    text: Optional[TextExpectation] = None
    audio: Optional[AudioExpectation] = None
    actions: List[Action] = field(default_factory=list)


# ------------------------------------------------------------
# Turn & Scenario
# ------------------------------------------------------------

@dataclass
class Turn:
    turn_id: int
    name: str
    description: str = ""
    content: Content
    transition: Transition
    expected: Expected


@dataclass
class Scenario:
    id: str
    name: str
    description: str
    version: str = "1.0"
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    turns: List[Turn] = field(default_factory=list)


# ============================================================
# 3. 剧本校验器（轻量级，仅做结构合法性检查）
# ============================================================

class ValidationError(Exception):
    pass


class ScenarioValidator:
    """
    对剧本做静态合法性校验。
    不检查音频文件是否存在（那是音频编辑子系统的职责）。
    """

    @staticmethod
    def validate(scenario: Scenario) -> None:
        errors: List[str] = []

        # 1. turn_id 连续递增
        for i, turn in enumerate(scenario.turns):
            if turn.turn_id != i:
                errors.append(f"Turn turn_id 必须连续递增：期望 {i}，实际 {turn.turn_id}")

        # 2. 必填字段
        for turn in scenario.turns:
            if not turn.content.base_clip_id:
                errors.append(f"Turn {turn.turn_id}: base_clip_id 不能为空")
            if not turn.content.text_ground_truth:
                errors.append(f"Turn {turn.turn_id}: text_ground_truth 不能为空")

            # Transition 合法性
            if turn.transition.type == "fixed":
                if turn.transition.fixed is None:
                    errors.append(f"Turn {turn.turn_id}: fixed transition 缺少 fixed 字段")
                else:
                    ft = turn.transition.fixed
                    if ft.mode == "silence" and ft.silence_duration_ms is None:
                        errors.append(f"Turn {turn.turn_id}: silence 模式必须提供 silence_duration_ms")
                    if ft.mode == "ambient" and (ft.ambient_clip_id is None or ft.ambient_duration_ms is None):
                        errors.append(f"Turn {turn.turn_id}: ambient 模式必须提供 clip_id 与 duration_ms")
            elif turn.transition.type == "reactive":
                if turn.transition.reactive is None:
                    errors.append(f"Turn {turn.turn_id}: reactive transition 缺少 reactive 字段")
                else:
                    rt = turn.transition.reactive
                    required_params = {
                        "after_model_speech": ["silence_threshold_ms"],
                        "after_model_speech_plus": ["silence_threshold_ms", "post_silence_wait_ms"],
                        "fixed_delay_interrupt": ["inject_after_model_start_ms"],
                        "immediate": [],
                    }
                    needed = required_params.get(rt.trigger, [])
                    for p in needed:
                        if p not in rt.params:
                            errors.append(f"Turn {turn.turn_id}: trigger '{rt.trigger}' 缺少参数 '{p}'")

            # Expected 一致性
            exp = turn.expected
            if exp.response_type == "none" and (exp.text or exp.audio):
                errors.append(f"Turn {turn.turn_id}: response_type=none 时不应有 text/audio 子字段")
            if exp.response_type in ("text", "both") and exp.text is None:
                errors.append(f"Turn {turn.turn_id}: response_type={exp.response_type} 时必须提供 text 字段")

        if errors:
            raise ValidationError("\n".join(errors))

    @staticmethod
    def validate_json(scenario_dict: Dict[str, Any]) -> Scenario:
        """
        从字典反序列化并校验。
        实际工程中可替换为 pydantic / marshmallow。
        """
        # 简易构造（生产环境建议使用更健壮的反序列化）
        global_cfg = scenario_dict.get("global", {})
        bg = global_cfg.get("background_noise")
        background_noise = BackgroundNoise(**bg) if bg else None
        global_config = GlobalConfig(
            sample_rate=global_cfg.get("sample_rate", 24000),
            initial_silence_ms=global_cfg.get("initial_silence_ms", 1000),
            background_noise=background_noise,
        )

        turns: List[Turn] = []
        for t in scenario_dict.get("turns", []):
            proc = Processing(**t["content"].get("processing", {}))
            content = Content(
                base_clip_id=t["content"]["base_clip_id"],
                text_ground_truth=t["content"]["text_ground_truth"],
                processing=proc,
            )

            trans_cfg = t["transition"]
            if trans_cfg["type"] == "fixed":
                fixed = FixedTransition(**trans_cfg.get("fixed", {}))
                transition = Transition(type="fixed", fixed=fixed)
            else:
                reactive = ReactiveTransition(**trans_cfg.get("reactive", {}))
                transition = Transition(type="reactive", reactive=reactive)

            exp_cfg = t["expected"]
            text_exp = None
            if "text" in exp_cfg:
                text_exp = TextExpectation(**exp_cfg["text"])
            audio_exp = None
            if "audio" in exp_cfg:
                audio_exp = AudioExpectation(**exp_cfg["audio"])
            actions = [Action(**a) for a in exp_cfg.get("actions", [])]
            expected = Expected(
                response_type=exp_cfg["response_type"],
                text=text_exp,
                audio=audio_exp,
                actions=actions,
            )

            turns.append(Turn(
                turn_id=t["turn_id"],
                name=t["name"],
                description=t.get("description", ""),
                content=content,
                transition=transition,
                expected=expected,
            ))

        scenario = Scenario(
            id=scenario_dict["id"],
            name=scenario_dict["name"],
            description=scenario_dict["description"],
            version=scenario_dict.get("version", "1.0"),
            global_config=global_config,
            turns=turns,
        )

        ScenarioValidator.validate(scenario)
        return scenario
