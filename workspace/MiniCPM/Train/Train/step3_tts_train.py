#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MiniCPM-o-4.5 端到端 TTS 微调训练脚本（语音到语音复述）
修正点：
  - 显式传入 audio_parts_list，确保 Processor 知道音频属于哪个 message
"""

import os
import sys
import importlib.util
from enum import Enum
import json
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from tqdm import tqdm
import numpy as np

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

# ========== 导入 VQ 编解码器 ==========
try:
    from vq_codec import VQCodec
except ImportError:
    vq_codec_path = os.path.join(SCRIPT_DIR, "vq_codec.py")
    if os.path.exists(vq_codec_path):
        spec_vq = importlib.util.spec_from_file_location("vq_codec", vq_codec_path)
        vq_codec_mod = importlib.util.module_from_spec(spec_vq)
        sys.modules["vq_codec"] = vq_codec_mod
        spec_vq.loader.exec_module(vq_codec_mod)
        VQCodec = vq_codec_mod.VQCodec
    else:
        raise ImportError("找不到 vq_codec.py，请将其放在项目根目录或 Train/ 目录下")

try:
    import librosa
except ImportError:
    raise ImportError("请安装 librosa: pip install librosa soundfile")


# ==================== 配置类 ====================

@dataclass
class TTSDataConfig:
    """TTS 数据配置"""
    task_dir: str = "data/tts"
    train_json: str = "train.json"
    val_json: Optional[str] = None
    system_prompt_file: Optional[str] = None
    audio_dir: str = "audio"
    audio_subdirs: Optional[List[str]] = None
    max_audio_length: int = 30
    max_target_audio_length: int = 30


@dataclass
class DataConfig:
    tts: TTSDataConfig = field(default_factory=TTSDataConfig)


@dataclass
class TrainConfig:
    model_name_or_path: str = "openbmb/MiniCPM-o-4_5"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    mode: str = "chat"

    freeze_llm: bool = False
    freeze_vision: bool = True
    freeze_audio: bool = True
    freeze_tts: bool = False
    unfreeze_llm_layers: Optional[List[int]] = None

    output_dir: str = "./minicpmo_tts_finetuned"
    data: DataConfig = field(default_factory=DataConfig)

    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    num_workers: int = 0
    max_text_length: int = 512

    tts_loss_weight: float = 1.0
    text_loss_weight: float = 0.1

    save_steps: int = 500
    log_steps: int = 10


def load_config(path: str = "config.yaml") -> TrainConfig:
    if not os.path.exists(path):
        print(f"[WARN] 未找到 {path}，使用默认配置")
        return TrainConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    training_cfg = raw.get("training", {})
    if not isinstance(training_cfg, dict):
        print(f"[ERROR] config.yaml 中 'training' 节点格式错误")
        training_cfg = {}

    top_fields = {}
    for k, v in raw.items():
        if k in ("data", "training"):
            continue
        if k in TrainConfig.__dataclass_fields__:
            top_fields[k] = v

    for k, v in training_cfg.items():
        if k in TrainConfig.__dataclass_fields__:
            top_fields[k] = v

    data_raw = raw.get("data", {})
    if not isinstance(data_raw, dict):
        data_raw = {}

    tts_raw = data_raw.get("tts", {})
    if not isinstance(tts_raw, dict):
        tts_raw = {}

    data_cfg = DataConfig(tts=TTSDataConfig(**tts_raw))

    float_fields = ["learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm", "tts_loss_weight", "text_loss_weight"]
    int_fields = ["batch_size", "gradient_accumulation_steps", "num_epochs", "num_workers", "max_text_length", "save_steps", "log_steps", "max_audio_length", "max_target_audio_length"]
    bool_fields = ["freeze_llm", "freeze_vision", "freeze_audio", "freeze_tts", "trust_remote_code"]

    for k in float_fields:
        if k in top_fields and isinstance(top_fields[k], str):
            top_fields[k] = float(top_fields[k])

    for k in int_fields:
        if k in top_fields and isinstance(top_fields[k], str):
            top_fields[k] = int(top_fields[k])

    for k in bool_fields:
        if k in top_fields and isinstance(top_fields[k], str):
            top_fields[k] = top_fields[k].lower() in ("true", "1", "yes", "on")

    return TrainConfig(data=data_cfg, **top_fields)


# ==================== TTS 数据集 ====================

class TTSDataset(Dataset):
    """
    TTS 数据集（语音到语音复述）
    关键修正：显式记录 audio_parts，告知 Processor 音频属于哪个 message
    """

    def __init__(self, task_cfg: TTSDataConfig, processor: MiniCPMOProcessor, config: TrainConfig, split: str = "train"):
        self.processor = processor
        self.config = config
        self.split = split
        self.tts_cfg = task_cfg

        self.task_dir = self._resolve_path(self.tts_cfg.task_dir)
        if not os.path.isdir(self.task_dir):
            raise FileNotFoundError(f"TTS 任务目录不存在: {self.task_dir}")

        json_path = self._resolve_path(self.tts_cfg.train_json, base=self.task_dir)
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.audio_search_dirs = self._build_audio_search_dirs()
        self._audio_path_cache = self._scan_audio_files()
        print(f"[TTSDataset] {split}: 找到 {len(self._audio_path_cache)} 个音频文件")

        self._global_system_prompt = self._load_system_prompt()
        print(f"[TTSDataset] {split}: 加载 {len(self.data)} 条样本 from {json_path}")

    def _resolve_path(self, path: str, base: Optional[str] = None) -> str:
        if not path:
            return ""
        if os.path.isabs(path):
            return path
        if base:
            candidate = os.path.join(base, path)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        candidate = os.path.join(SCRIPT_DIR, path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
        return os.path.abspath(os.path.join(PROJECT_ROOT, path))

    def _build_audio_search_dirs(self) -> List[str]:
        audio_root = os.path.join(self.task_dir, self.tts_cfg.audio_dir or "audio")
        subdirs = getattr(self.tts_cfg, "audio_subdirs", None)
        if not subdirs:
            return [audio_root]
        if isinstance(subdirs, str):
            subdirs = [subdirs]
        search_dirs = []
        for sub in subdirs:
            full_path = os.path.join(audio_root, sub)
            if os.path.isdir(full_path):
                search_dirs.append(full_path)
        if not search_dirs:
            return [audio_root]
        return search_dirs

    def _scan_audio_files(self) -> Dict[str, str]:
        cache = {}
        for search_dir in self.audio_search_dirs:
            for root, _, files in os.walk(search_dir):
                for fname in files:
                    if fname.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
                        if fname not in cache:
                            cache[fname] = os.path.join(root, fname)
        return cache

    def _find_audio_path(self, audio_name: str) -> str:
        if os.path.isabs(audio_name):
            if not os.path.exists(audio_name):
                raise FileNotFoundError(f"音频文件不存在: {audio_name}")
            return audio_name
        if os.path.dirname(audio_name):
            candidate = os.path.join(self.task_dir, audio_name)
            if os.path.exists(candidate):
                return candidate
            audio_root = os.path.join(self.task_dir, self.tts_cfg.audio_dir or "audio")
            candidate = os.path.join(audio_root, audio_name)
            if os.path.exists(candidate):
                return candidate
            raise FileNotFoundError(f"音频文件不存在: {audio_name}")
        if audio_name in self._audio_path_cache:
            return self._audio_path_cache[audio_name]
        raise FileNotFoundError(f"音频文件未找到: {audio_name}")

    def _load_system_prompt(self) -> Optional[str]:
        file_path = self.tts_cfg.system_prompt_file
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

        audio_path = self._find_audio_path(item["audio"])
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)

        max_samples_input = int(self.tts_cfg.max_audio_length * 16000)
        max_samples_target = int(self.tts_cfg.max_target_audio_length * 16000)

        input_waveform = waveform[:max_samples_input] if len(waveform) > max_samples_input else waveform
        target_waveform = waveform[:max_samples_target] if len(waveform) > max_samples_target else waveform

        system_text = (
            item.get("system_prompt")
            or self._global_system_prompt
            or "请复述用户提供的语音内容。"
        )

        # 构建对话：user 的 content 是 list，包含音频 + 文本
        msgs = [
            {"role": "system", "content": system_text},                           # index 0
            {"role": "user", "content": [input_waveform, "请复述这段语音。"]},   # index 1，包含音频
            {"role": "assistant", "content": f"||<|<|tts_bos|>{item['text']}<<||<|<|tts_eos|>"}  # index 2
        ]

        # 关键修正：显式记录音频属于第 1 个 message（user）
        # 这样 Processor 才能正确将音频 embedding 替换到 <audio>./</audio> 的位置
        audio_parts = [1]

        prompt = self.processor.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {
            "prompt": prompt,
            "audio": input_waveform,
            "target_audio": target_waveform.copy(),
            "text": item["text"],
            "audio_path": audio_path,
            "audio_parts": audio_parts,  # 新增：显式携带
        }


# ==================== DataCollator（修正版） ====================

class TTSDataCollator:
    """
    TTS 数据整理器：
      - 显式传入 audio_parts_list，确保 Processor 正确插入音频 embedding
      - 构建 labels 与 TTS 目标 VQ tokens
    """

    def __init__(self, processor: MiniCPMOProcessor, device: str, vq_codec: VQCodec, max_length: int = 512):
        self.processor = processor
        self.device = device
        self.vq_codec = vq_codec
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        prompts = [item["prompt"] for item in batch]
        input_audios = [item["audio"] for item in batch]
        target_audios = [item["target_audio"] for item in batch]
        audio_parts_list = [item["audio_parts"] for item in batch]  # 新增：收集 audio_parts

        # 关键修正：显式传入 audio_parts_list
        inputs = self.processor(
            prompts,
            [[] for _ in batch],                      # images
            [[audio] for audio in input_audios],      # audios
            audio_parts_list,                           # 新增：音频所属 message 索引
            return_tensors="pt",
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        # 定位 <|tts_bos|> 和 <|tts_eos|>
        tts_bos_token_id = self.processor.tokenizer.convert_tokens_to_ids("||<|<|tts_bos|>")
        tts_eos_token_id = self.processor.tokenizer.convert_tokens_to_ids("||<|<|tts_eos|>")

        tts_bounds = []
        for b in range(input_ids.shape[0]):
            ids = input_ids[b].tolist()
            try:
                bos_idx = ids.index(tts_bos_token_id)
                eos_candidates = [i for i, x in enumerate(ids) if x == tts_eos_token_id and i > bos_idx]
                eos_idx = eos_candidates[0] if eos_candidates else -1
                if eos_idx == -1:
                    actual_len = attention_mask[b].sum().item()
                    eos_idx = actual_len
            except ValueError:
                bos_idx = -1
                eos_idx = -1
            tts_bounds.append((bos_idx, eos_idx))

        # 编码目标音频为 VQ tokens
        target_vq_tokens = []
        for audio in target_audios:
            tokens = self.vq_codec.encode(audio)
            target_vq_tokens.append(tokens)

        # 文本 labels（Causal LM）
        labels = input_ids.clone().long()
        pad_token_id = self.processor.tokenizer.pad_token_id or 0
        labels[labels == pad_token_id] = -100

        model_inputs = {
            "input_ids": input_ids.to(self.device),
            "position_ids": torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0).expand(input_ids.shape[0], -1),
            "audio_features": inputs.get("audio_features"),
            "audio_feature_lens": inputs.get("audio_feature_lens"),
            "image_bound": inputs.get("image_bound"),
            "audio_bounds": inputs.get("audio_bounds"),
            "spk_bounds": inputs.get("spk_bounds"),
            "pixel_values": inputs.get("pixel_values"),
            "tgt_sizes": inputs.get("tgt_sizes"),
            "labels": labels.to(self.device),
            "tts_bounds": tts_bounds,
            "target_vq_tokens": target_vq_tokens,
        }

        return {k: v for k, v in model_inputs.items() if v is not None}


# ==================== 冻结策略 ====================

def apply_freeze_strategy(model: MiniCPMO, cfg: TrainConfig):
    print("\n" + "=" * 60)
    print("应用冻结策略")
    print("=" * 60)

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

    if hasattr(model, "vpm") and model.vpm is not None:
        freeze = cfg.freeze_vision
        for p in model.vpm.parameters():
            p.requires_grad = not freeze
        for p in model.resampler.parameters():
            p.requires_grad = not freeze
        print(f"  → Vision Encoder & Resampler         {'冻结' if freeze else '可训练'}")

    if hasattr(model, "apm") and model.apm is not None:
        freeze = cfg.freeze_audio
        for p in model.apm.parameters():
            p.requires_grad = not freeze
        for p in model.audio_projection_layer.parameters():
            p.requires_grad = not freeze
        print(f"  → Audio Encoder & Projection         {'冻结' if freeze else '可训练'}")

    if hasattr(model, "tts") and model.tts is not None:
        freeze = cfg.freeze_tts
        for p in model.tts.parameters():
            p.requires_grad = not freeze
        print(f"  → TTS (MiniCPMTTS)                   {'冻结' if freeze else '可训练'}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Params] Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.2f}%)")


# ==================== TTS Loss 计算 ====================

def compute_tts_loss(model: MiniCPMO, batch: Dict, llm_hidden_states: torch.Tensor) -> torch.Tensor:
    device = llm_hidden_states.device
    tts = model.tts
    tts_bounds = batch["tts_bounds"]
    target_vq_tokens = batch["target_vq_tokens"]

    losses = []
    valid_samples = 0

    for b in range(len(tts_bounds)):
        bos_idx, eos_idx = tts_bounds[b]
        if bos_idx < 0 or eos_idx <= bos_idx:
            continue

        sample_input_ids = batch["input_ids"][b]
        llm_tokens = sample_input_ids[bos_idx:eos_idx]
        llm_hidden = llm_hidden_states[b, bos_idx:eos_idx, :]

        if llm_tokens.numel() == 0 or llm_hidden.shape[0] == 0:
            continue

        llm_embeds = tts.emb_text(llm_tokens)

        proj_hidden = tts.projector_semantic(llm_hidden)
        if getattr(tts.config, "normalize_projected_hidden", False):
            proj_hidden = F.normalize(proj_hidden, p=2, dim=-1)

        tts_embeds = llm_embeds + proj_hidden

        spk_embeds = torch.zeros(0, tts.config.hidden_size, device=device, dtype=tts_embeds.dtype)

        text_eos_id = getattr(tts.config, "text_eos_token_id", 2)
        audio_bos_id = getattr(tts, "audio_bos_token_id", 1)

        text_eos_embed = tts.emb_text(torch.tensor([text_eos_id], device=device, dtype=torch.long))
        audio_bos_embed = tts.emb_text(torch.tensor([audio_bos_id], device=device, dtype=torch.long))

        condition = torch.cat([spk_embeds, tts_embeds, text_eos_embed, audio_bos_embed], dim=0).unsqueeze(0)

        target_tokens = torch.from_numpy(target_vq_tokens[b]).long().to(device)

        if target_tokens.dim() == 1:
            target_tokens = target_tokens.unsqueeze(1)
        if target_tokens.shape[1] < tts.num_vq:
            target_tokens = target_tokens.repeat(1, tts.num_vq)
        elif target_tokens.shape[1] > tts.num_vq:
            target_tokens = target_tokens[:, :tts.num_vq]

        seq_len = target_tokens.shape[0]
        if seq_len == 0:
            continue

        audio_input = target_tokens[:-1, :].unsqueeze(0) if seq_len > 1 else torch.zeros(1, 0, tts.num_vq, device=device, dtype=torch.long)
        audio_embeds = []
        for q in range(tts.num_vq):
            audio_embeds.append(tts.emb_code[q](audio_input[:, :, q]))
        audio_embeds = torch.stack(audio_embeds, dim=-1).sum(dim=-1) if seq_len > 1 else torch.zeros(1, 0, tts.config.hidden_size, device=device)

        full_embeds = torch.cat([condition, audio_embeds], dim=1)
        position_ids = torch.arange(full_embeds.shape[1], device=device).unsqueeze(0)

        tts_outputs = tts.model(
            inputs_embeds=full_embeds,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=False,
        )

        audio_hidden = tts_outputs.last_hidden_state[:, condition.shape[1] - 1:, :]

        loss_b = 0
        for q in range(tts.num_vq):
            logits_q = tts.head_code[q](audio_hidden)
            targets_q = target_tokens[:, q].unsqueeze(0)
            loss_b += F.cross_entropy(
                logits_q.reshape(-1, logits_q.size(-1)),
                targets_q.reshape(-1),
                ignore_index=-100,
            )

        losses.append(loss_b / tts.num_vq)
        valid_samples += 1

    if not losses:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean()


# ==================== 训练循环 ====================

def train():
    cfg = load_config("config.yaml")
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, cfg.torch_dtype, torch.bfloat16)

    # 1. 初始化 VQ 编解码器
    print("=" * 60)
    print("正在初始化 VQ 编解码器...")
    print("=" * 60)
    vq_codec = VQCodec(
        model_dir=cfg.model_name_or_path,
        device=str(device),
        s3tokenizer_name="speech_tokenizer_v2_25hz",
    )

    # 2. 加载模型
    print("=" * 60)
    print("正在加载 MiniCPM-o-4.5...")
    print("=" * 60)

    model = MiniCPMO.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=dtype,
    ).to(device)

    model.config.stream_input = False

    if hasattr(model, "set_mode"):
        mode_map = {"chat": ProcessorMode.CHAT, "streaming": ProcessorMode.STREAMING, "duplex": ProcessorMode.DUPLEX}
        model.set_mode(mode_map.get(cfg.mode, ProcessorMode.CHAT))

    # 3. 加载 Processor
    processor = MiniCPMOProcessor.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
    )

    # 4. 冻结策略
    apply_freeze_strategy(model, cfg)

    # 5. 数据集
    train_dataset = TTSDataset(
        cfg.data.tts,
        processor,
        cfg,
        split="train",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=TTSDataCollator(processor, str(device), vq_codec, max_length=cfg.max_text_length),
    )

    # 6. 优化器与调度器
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

    # 7. 混合精度
    use_amp = (cfg.device == "cuda" and cfg.torch_dtype == "float16")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 8. 训练循环
    model.train()
    global_step = 0
    total_loss = 0
    total_text_loss = 0
    total_tts_loss = 0

    for epoch in range(cfg.num_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        epoch_loss = 0.0
        epoch_text_loss = 0.0
        epoch_tts_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(progress_bar):
            # --- LLM Forward ---
            with torch.cuda.amp.autocast(dtype=dtype) if cfg.device == "cuda" else torch.nullcontext():
                outputs = model(batch, return_dict=True, output_hidden_states=True)
                llm_logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                llm_hidden_states = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else None

                # 文本 Loss（Causal LM）
                labels = batch["labels"]
                shift_logits = llm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().long()

                text_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                # TTS Loss
                tts_loss = torch.tensor(0.0, device=device)
                if llm_hidden_states is not None and cfg.tts_loss_weight > 0:
                    tts_loss = compute_tts_loss(model, batch, llm_hidden_states)

                # 总 Loss
                loss = text_loss * cfg.text_loss_weight + tts_loss * cfg.tts_loss_weight
                loss = loss / cfg.gradient_accumulation_steps

            # --- 反向传播 ---
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # --- 记录 ---
            raw_loss = loss.item() * cfg.gradient_accumulation_steps
            raw_text = text_loss.item()
            raw_tts = tts_loss.item()

            total_loss += raw_loss
            total_text_loss += raw_text
            total_tts_loss += raw_tts
            epoch_loss += raw_loss
            epoch_text_loss += raw_text
            epoch_tts_loss += raw_tts
            num_batches += 1

            progress_bar.set_postfix({
                "loss": f"{raw_loss:.4f}",
                "text": f"{raw_text:.4f}",
                "tts": f"{raw_tts:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}" if global_step > 0 else f"{cfg.learning_rate:.2e}",
            })

            # --- 梯度更新 ---
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

                if global_step % cfg.log_steps == 0:
                    avg_loss = total_loss / cfg.log_steps
                    avg_text = total_text_loss / cfg.log_steps
                    avg_tts = total_tts_loss / cfg.log_steps
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"\n[Step {global_step}] loss={avg_loss:.4f} text={avg_text:.4f} tts={avg_tts:.4f} lr={current_lr:.2e}")
                    total_loss = 0
                    total_text_loss = 0
                    total_tts_loss = 0

                if global_step % cfg.save_steps == 0:
                    save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    processor.save_pretrained(save_path)
                    print(f"[Save] 检查点已保存: {save_path}")

        # Epoch 总结
        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_text = epoch_text_loss / num_batches
        avg_epoch_tts = epoch_tts_loss / num_batches
        print(f"\n{'='*50}")
        print(f"[Epoch {epoch+1}/{cfg.num_epochs}] loss={avg_epoch_loss:.4f} text={avg_epoch_text:.4f} tts={avg_epoch_tts:.4f}")
        print(f"{'='*50}\n")

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
