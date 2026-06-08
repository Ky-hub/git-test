#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 2: ASR 微调训练脚本
支持：
  - data 目录在 Train/ 下（data/asr/audio/xxx.wav）
  - config.yaml 嵌套配置（data.asr.task_dir / audio_subdirs）
  - system prompt 从外部文本文件读取（data/asr/system_prompt.txt）
  - 多子目录音频自动扫描
  - 防御性 YAML 类型转换（防止 "1e-5" 字符串错误）
  - 命令行指定显卡：CUDA_VISIBLE_DEVICES=0 python step2_asr_train.py
"""

import os
import sys
import importlib.util
from enum import Enum
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
from tqdm import tqdm

# ========== 本地定义 ProcessorMode（兼容精简版模型文件） ==========
class ProcessorMode(Enum):
    CHAT = "chat"
    STREAMING = "streaming"
    DUPLEX = "duplex"

# ========== 路径适配 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# ========== 加载 MiniCPMO45/utils.py 并补充缺失函数（不改原文件） ==========
utils_path = os.path.join(PROJECT_ROOT, "MiniCPMO45", "utils.py")
if not os.path.exists(utils_path):
    raise FileNotFoundError(f"找不到 utils.py: {utils_path}")

spec = importlib.util.spec_from_file_location("MiniCPMO45.utils", utils_path)
utils_mod = importlib.util.module_from_spec(spec)
sys.modules["MiniCPMO45.utils"] = utils_mod
spec.loader.exec_module(utils_mod)

if not hasattr(utils_mod, "normalize_content"):
    utils_mod.normalize_content = lambda x: (
        [x] if isinstance(x, str) else (x if isinstance(x, list) else [x])
    )
    print("[Patch] 已补充 normalize_content 到 MiniCPMO45.utils")

# ========== 导入模型 ==========
from MiniCPMO45.modeling_minicpmo import MiniCPMO
from MiniCPMO45.processing_minicpmo import MiniCPMOProcessor

try:
    import librosa
except ImportError:
    raise ImportError("请安装 librosa: pip install librosa soundfile")


# ==================== 配置类（嵌套结构，支持后续扩展） ====================

@dataclass
class ASRDataConfig:
    """ASR 数据配置（data 目录分级）"""
    task_dir: str = "data/asr"
    train_json: str = "train.json"
    val_json: Optional[str] = None
    system_prompt_file: Optional[str] = None
    audio_dir: str = "audio"
    audio_subdirs: Optional[List[str]] = None
    max_audio_length: int = 30


@dataclass
class DataConfig:
    """数据根节点，按任务分级。后续扩展在此加同级字段"""
    asr: ASRDataConfig = field(default_factory=ASRDataConfig)


@dataclass
class TrainConfig:
    """全局训练配置"""
    model_name_or_path: str = "openbmb/MiniCPM-o-4_5"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    mode: str = "chat"

    freeze_llm: bool = False
    freeze_vision: bool = True
    freeze_audio: bool = True
    freeze_tts: bool = True
    unfreeze_llm_layers: Optional[List[int]] = None

    output_dir: str = "./minicpmo_asr_finetuned"
    data: DataConfig = field(default_factory=DataConfig)

    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    num_workers: int = 4
    max_text_length: int = 256

    save_steps: int = 500
    log_steps: int = 10


def load_config(path: str = "config.yaml") -> TrainConfig:
    """解析 YAML，支持嵌套 data.asr / training 节点，带错误诊断和类型转换"""
    if not os.path.exists(path):
        print(f"[WARN] 未找到 {path}，使用默认配置")
        return TrainConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # 诊断：检查 training 节点
    training_cfg = raw.get("training", {})
    if not isinstance(training_cfg, dict):
        print(f"[ERROR] config.yaml 中 'training' 节点格式错误，类型为 {type(training_cfg)}")
        print(f"[ERROR] 请检查 YAML 缩进，确保 training 下方有正确的键值对")
        training_cfg = {}

    # 1. 提取顶层字段（排除 data / training）
    top_fields = {}
    for k, v in raw.items():
        if k in ("data", "training"):
            continue
        if k in TrainConfig.__dataclass_fields__:
            top_fields[k] = v

    # 2. 扁平化 training 节点到顶层
    for k, v in training_cfg.items():
        if k in TrainConfig.__dataclass_fields__:
            top_fields[k] = v

    # 3. 构建嵌套 DataConfig
    data_raw = raw.get("data", {})
    if not isinstance(data_raw, dict):
        print(f"[ERROR] config.yaml 中 'data' 节点格式错误")
        data_raw = {}

    asr_raw = data_raw.get("asr", {})
    if not isinstance(asr_raw, dict):
        print(f"[ERROR] config.yaml 中 'data.asr' 节点格式错误")
        asr_raw = {}

    data_cfg = DataConfig(asr=ASRDataConfig(**asr_raw))

    # 4. 防御性类型转换（防止 YAML 解析为字符串）
    float_fields = ["learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm"]
    int_fields = ["batch_size", "gradient_accumulation_steps", "num_epochs",
                  "num_workers", "max_text_length", "save_steps", "log_steps"]
    bool_fields = ["freeze_llm", "freeze_vision", "freeze_audio", "freeze_tts", "trust_remote_code"]

    for k in float_fields:
        if k in top_fields and isinstance(top_fields[k], str):
            try:
                top_fields[k] = float(top_fields[k])
            except ValueError as e:
                raise ValueError(f"配置项 '{k}' 的值 '{top_fields[k]}' 无法转换为浮点数，请检查 YAML 格式") from e

    for k in int_fields:
        if k in top_fields and isinstance(top_fields[k], str):
            try:
                top_fields[k] = int(top_fields[k])
            except ValueError as e:
                raise ValueError(f"配置项 '{k}' 的值 '{top_fields[k]}' 无法转换为整数，请检查 YAML 格式") from e

    for k in bool_fields:
        if k in top_fields and isinstance(top_fields[k], str):
            top_fields[k] = top_fields[k].lower() in ("true", "1", "yes", "on")

    return TrainConfig(data=data_cfg, **top_fields)


# ==================== ASR 数据集（data 在 Train/ 下，支持多子目录） ====================

class ASRDataset(Dataset):
    """
    ASR 数据集：data 目录位于 Train/ 下
    支持多子目录音频自动扫描
    system prompt 优先级：单条数据 > 外部文本文件 > 默认
    """

    def __init__(
        self,
        task_cfg: ASRDataConfig,
        processor: MiniCPMOProcessor,
        config: TrainConfig,
        split: str = "train",
    ):
        self.processor = processor
        self.config = config
        self.split = split
        self.asr_cfg = task_cfg

        # 解析任务根目录（优先相对于 SCRIPT_DIR = Train/）
        self.task_dir = self._resolve_path(self.asr_cfg.task_dir)
        if not os.path.isdir(self.task_dir):
            raise FileNotFoundError(f"ASR 任务目录不存在: {self.task_dir}")

        # 加载 JSON
        json_path = self._resolve_path(self.asr_cfg.train_json, base=self.task_dir)
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # 构建音频搜索路径
        self.audio_search_dirs = self._build_audio_search_dirs()
        self._audio_path_cache = self._scan_audio_files()
        print(f"[ASRDataset] {split}: 找到 {len(self._audio_path_cache)} 个音频文件")

        # 加载 system prompt
        self._global_system_prompt = self._load_system_prompt()
        if self._global_system_prompt:
            print(f"[ASRDataset] {split}: 从文件加载 system prompt ({len(self._global_system_prompt)} 字符)")
        else:
            print(f"[ASRDataset] {split}: 使用默认 system prompt")

        print(f"[ASRDataset] {split}: 加载 {len(self.data)} 条样本 from {json_path}")
        print(f"[ASRDataset] {split}: 音频目录: {self.audio_search_dirs}")

    def _resolve_path(self, path: str, base: Optional[str] = None) -> str:
        """路径解析：优先相对于 SCRIPT_DIR（Train/ 目录）"""
        if not path:
            return ""
        if os.path.isabs(path):
            return path

        # 1. 优先相对于传入的 base
        if base:
            candidate = os.path.join(base, path)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        # 2. 其次相对于 SCRIPT_DIR（脚本所在目录 = Train/）
        candidate = os.path.join(SCRIPT_DIR, path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

        # 3. 最后相对于项目根目录
        return os.path.abspath(os.path.join(PROJECT_ROOT, path))

    def _build_audio_search_dirs(self) -> List[str]:
        """构建音频搜索目录列表"""
        audio_root = os.path.join(self.task_dir, self.asr_cfg.audio_dir or "audio")

        subdirs = getattr(self.asr_cfg, "audio_subdirs", None)
        if not subdirs:
            return [audio_root]

        if isinstance(subdirs, str):
            subdirs = [subdirs]

        search_dirs = []
        for sub in subdirs:
            full_path = os.path.join(audio_root, sub)
            if os.path.isdir(full_path):
                search_dirs.append(full_path)
            else:
                print(f"[WARN] 音频子目录不存在，已跳过: {full_path}")

        if not search_dirs:
            print(f"[WARN] 所有子目录无效，回退到: {audio_root}")
            return [audio_root]

        return search_dirs

    def _scan_audio_files(self) -> Dict[str, str]:
        """预扫描所有子目录，建立 {文件名: 完整路径} 映射"""
        cache = {}
        for search_dir in self.audio_search_dirs:
            for root, _, files in os.walk(search_dir):
                for fname in files:
                    if fname.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
                        if fname not in cache:
                            cache[fname] = os.path.join(root, fname)
        return cache

    def _find_audio_path(self, audio_name: str) -> str:
        """
        查找音频文件：
        1. 绝对路径 -> 直接返回
        2. 含子目录（如 "batch_01/001.wav"）-> 拼接
        3. 纯文件名 -> 查缓存
        """
        # 情况 1：绝对路径
        if os.path.isabs(audio_name):
            if not os.path.exists(audio_name):
                raise FileNotFoundError(f"音频文件不存在: {audio_name}")
            return audio_name

        # 情况 2：含子目录（如 "batch_01/001.wav"）
        if os.path.dirname(audio_name):
            candidate = os.path.join(self.task_dir, audio_name)
            if os.path.exists(candidate):
                return candidate
            audio_root = os.path.join(self.task_dir, self.asr_cfg.audio_dir or "audio")
            candidate = os.path.join(audio_root, audio_name)
            if os.path.exists(candidate):
                return candidate
            raise FileNotFoundError(f"音频文件不存在: {audio_name} (尝试: {candidate})")

        # 情况 3：纯文件名，查缓存
        if audio_name in self._audio_path_cache:
            return self._audio_path_cache[audio_name]

        raise FileNotFoundError(
            f"音频文件未找到: {audio_name}\n"
            f"搜索目录: {self.audio_search_dirs}\n"
            f"提示：请确认文件名正确，或改用 '子目录/文件名.wav' 格式"
        )

    def _load_system_prompt(self) -> Optional[str]:
        """从 asr_cfg.system_prompt_file 读取（相对于 task_dir）"""
        file_path = self.asr_cfg.system_prompt_file
        if not file_path:
            return None

        file_path = self._resolve_path(file_path, base=self.task_dir)
        if not os.path.exists(file_path):
            print(f"[WARN] system_prompt_file 不存在: {file_path}")
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content if content else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        # 1. 查找音频（支持多子目录）
        audio_path = self._find_audio_path(item["audio"])
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)

        # 截断
        max_samples = int(self.asr_cfg.max_audio_length * 16000)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        # 2. 构建对话（content 必须是字符串，不能放 list/array）
        msgs = []

        # system prompt
        system_text = (
            item.get("system_prompt")
            or self._global_system_prompt
            or "You are a helpful assistant. You can accept audio and text input and output voice and text."
        )
        msgs.append({"role": "system", "content": system_text})

        # user：content 只用纯文本，音频单独返回给 collator
        msgs.append({
            "role": "user",
            "content": "请转录这段音频的内容。"   # ← 纯字符串，不放 list
        })

        # assistant：正确转录文本（label）
        msgs.append({
            "role": "assistant",
            "content": item["text"]
        })

        # 3. 应用 chat template（现在 content 全是字符串，不会报错）
        prompt = self.processor.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {
            "prompt": prompt,           # 纯文本 prompt 字符串
            "audio": waveform,          # 音频 numpy array，在 collator 中处理
            "text": item["text"],
            "audio_path": audio_path,
        }


# ==================== DataCollator ====================

class ASRDataCollator:
    """
    ASR 数据整理器：
      - 调用 Processor 将文本+音频转为模型输入
      - 构建 labels（Causal LM 标准 shift，padding 设为 -100）
    """

    def __init__(self, processor: MiniCPMOProcessor, device: str, max_length: int = 512):
        self.processor = processor
        self.device = device
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        prompts = [item["prompt"] for item in batch]
        audios = [item["audio"] for item in batch]

        # Processor 编码
        inputs = self.processor(
            prompts,
            [[] for _ in batch],
            [[audio] for audio in audios],
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        # labels：padding 设为 -100
        labels = input_ids.clone()
        pad_token_id = self.processor.tokenizer.pad_token_id or 0
        labels[labels == pad_token_id] = -100

        # 组装模型输入
        model_inputs = {
            "input_ids": input_ids.to(self.device),
            "position_ids": torch.arange(input_ids.shape[1], device=self.device)
                              .unsqueeze(0)
                              .expand(input_ids.shape[0], -1),
            "audio_features": inputs.get("audio_features"),
            "audio_feature_lens": inputs.get("audio_feature_lens"),
            "image_bound": inputs.get("image_bound"),
            "audio_bounds": inputs.get("audio_bounds"),
            "spk_bounds": inputs.get("spk_bounds"),
            "pixel_values": inputs.get("pixel_values"),
            "tgt_sizes": inputs.get("tgt_sizes"),
            "labels": labels.to(self.device),
        }

        return {k: v for k, v in model_inputs.items() if v is not None}


# ==================== 冻结策略 ====================

def apply_freeze_strategy(model: MiniCPMO, cfg: TrainConfig):
    """根据配置冻结/解冻参数"""
    print("\n" + "=" * 60)
    print("应用冻结策略")
    print("=" * 60)

    # LLM
    if hasattr(model, "llm") and model.llm is not None:
        if cfg.freeze_llm:
            for p in model.llm.parameters():
                p.requires_grad = False
            if cfg.unfreeze_llm_layers:
                num_layers = len(model.llm.model.layers)
                for idx in cfg.unfreeze_llm_layers:
                    real_idx = idx if idx >= 0 else num_layers + idx
                    if 0 <= real_idx < num_layers:
                        for p in model.llm.model.layers[real_idx].parameters():
                            p.requires_grad = True
                        print(f"  → LLM Layer {real_idx:>3} ({idx:>+4})  已解冻")
                for p in model.llm.lm_head.parameters():
                    p.requires_grad = True
                for p in model.llm.model.embed_tokens.parameters():
                    p.requires_grad = True
                print("  → LLM lm_head / embed_tokens        已解冻")
            else:
                print("  → LLM 全部冻结")
        else:
            print("  → LLM 全部可训练")

    # Vision
    if hasattr(model, "vpm") and model.vpm is not None:
        freeze = cfg.freeze_vision
        for p in model.vpm.parameters():
            p.requires_grad = not freeze
        for p in model.resampler.parameters():
            p.requires_grad = not freeze
        print(f"  → Vision Encoder & Resampler         {'冻结' if freeze else '可训练'}")

    # Audio
    if hasattr(model, "apm") and model.apm is not None:
        freeze = cfg.freeze_audio
        for p in model.apm.parameters():
            p.requires_grad = not freeze
        for p in model.audio_projection_layer.parameters():
            p.requires_grad = not freeze
        print(f"  → Audio Encoder & Projection         {'冻结' if freeze else '可训练'}")

    # TTS
    if hasattr(model, "tts") and model.tts is not None:
        freeze = cfg.freeze_tts
        for p in model.tts.parameters():
            p.requires_grad = not freeze
        print(f"  → TTS (MiniCPMTTS)                   {'冻结' if freeze else '可训练'}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Params] Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.2f}%)")


# ==================== 训练循环 ====================

def train():
    cfg = load_config("config.yaml")
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, cfg.torch_dtype, torch.bfloat16)

    # 1. 加载模型
    print("=" * 60)
    print("正在加载模型...")
    print("=" * 60)

    model = MiniCPMO.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=dtype,
    ).to(device)
  
    
    model.config.stream_input = False

    if hasattr(model, "set_mode"):
        mode_map = {
            "chat": ProcessorMode.CHAT,
            "streaming": ProcessorMode.STREAMING,
            "duplex": ProcessorMode.DUPLEX,
        }
        model.set_mode(mode_map.get(cfg.mode, ProcessorMode.CHAT))

    # 2. 加载 Processor
    processor = MiniCPMOProcessor.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
    )

    # 3. 冻结策略
    apply_freeze_strategy(model, cfg)

    # 4. 数据集
    train_dataset = ASRDataset(
        cfg.data.asr,
        processor,
        cfg,
        split="train",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=ASRDataCollator(processor, str(device), max_length=cfg.max_text_length),
    )

    # 5. 优化器与调度器
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    total_steps = len(train_loader) * cfg.num_epochs // cfg.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps,
    )

    # 6. 混合精度
    use_amp = (cfg.device == "cuda" and cfg.torch_dtype == "float16")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 7. 训练循环
    model.train()
    global_step = 0
    total_loss = 0

    for epoch in range(cfg.num_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")

        for step, batch in enumerate(progress_bar):
            with torch.cuda.amp.autocast(dtype=dtype) if cfg.device == "cuda" else torch.nullcontext():
                outputs = model(batch, return_dict=True)

                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                labels = batch["labels"]

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss / cfg.gradient_accumulation_steps

            # 反向传播
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * cfg.gradient_accumulation_steps

            # 梯度更新
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # 日志
                if global_step % cfg.log_steps == 0:
                    avg_loss = total_loss / cfg.log_steps
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                    })
                    total_loss = 0

                # 保存检查点
                if global_step % cfg.save_steps == 0:
                    save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    processor.save_pretrained(save_path)
                    print(f"\n[Save] 检查点已保存: {save_path}")

    # 最终保存
    final_path = os.path.join(cfg.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\n[Done] 最终模型已保存: {final_path}")


import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    train()
