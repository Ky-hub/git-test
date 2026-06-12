#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MiniCPM-O-4.5 TTS 端到端训练脚本（调试模式：不优化 + 每batch重建检查）
"""

import os
import sys
import json
import math
import logging
import multiprocessing
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO, ProcessorMode
from MiniCPMO45.processing_minicpmo import MiniCPMOProcessor

try:
    from vq_codec import VQCodec
except ImportError:
    raise ImportError("未找到 vq_codec.py")

try:
    import librosa
    import soundfile as sf
except ImportError:
    raise ImportError("请安装依赖: pip install librosa soundfile")

try:
    from scipy import signal
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ==================== 配置类（新增 debug_no_optimize） ====================

@dataclass
class TTSDataConfig:
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

    # 调试开关
    debug_mode: bool = False
    debug_no_optimize: bool = False        # 新增：True = 不执行 optimizer.step()，只做前向+重建检查
    debug_reconstruct: bool = False
    debug_reconstruct_steps: int = 1      # 每 N 个 batch 触发一次重建（不依赖 global_step）
    tts_loss_weight: float = 1.0
    text_loss_weight: float = 0.1

    save_steps: int = 500
    log_steps: int = 10


def load_config(path: str = "config.yaml") -> TrainConfig:
    if not os.path.exists(path):
        return TrainConfig()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    training_cfg = raw.get("training", {}) or {}
    top_fields = {}
    for k, v in {**raw, **training_cfg}.items():
        if k in TrainConfig.__dataclass_fields__:
            top_fields[k] = v
    data_raw = raw.get("data", {})
    tts_raw = data_raw.get("tts", {}) if isinstance(data_raw, dict) else {}
    data_cfg = DataConfig(tts=TTSDataConfig(**tts_raw))
    # 类型转换
    for k in ["learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm", "tts_loss_weight", "text_loss_weight"]:
        if k in top_fields and isinstance(top_fields[k], str):
            top_fields[k] = float(top_fields[k])
    for k in ["batch_size", "gradient_accumulation_steps", "num_epochs", "num_workers", "max_text_length", "save_steps", "log_steps", "max_audio_length", "max_target_audio_length", "debug_reconstruct_steps"]:
        if k in top_fields and isinstance(top_fields[k], str):
            top_fields[k] = int(top_fields[k])
    for k in ["freeze_llm", "freeze_vision", "freeze_audio", "freeze_tts", "trust_remote_code", "debug_mode", "debug_reconstruct", "debug_no_optimize"]:
        if k in top_fields and isinstance(top_fields[k], str):
            top_fields[k] = top_fields[k].lower() in ("true", "1", "yes", "on")
    return TrainConfig(data=data_cfg, **top_fields)


# ==================== 数据集（与之前一致） ====================

class TTSDataset(Dataset):
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
        print(f"[TTSDataset] {split}: 加载 {len(self.data)} 条样本")

    def _resolve_path(self, path: str, base: Optional[str] = None) -> str:
        if not path: return ""
        if os.path.isabs(path): return path
        if base:
            c = os.path.join(base, path)
            if os.path.exists(c): return os.path.abspath(c)
        c = os.path.join(SCRIPT_DIR, path)
        if os.path.exists(c): return os.path.abspath(c)
        return os.path.abspath(os.path.join(PROJECT_ROOT, path))

    def _build_audio_search_dirs(self) -> List[str]:
        audio_root = os.path.join(self.task_dir, self.tts_cfg.audio_dir or "audio")
        subdirs = getattr(self.tts_cfg, "audio_subdirs", None)
        if not subdirs: return [audio_root]
        if isinstance(subdirs, str): subdirs = [subdirs]
        return [os.path.join(audio_root, s) for s in subdirs if os.path.isdir(os.path.join(audio_root, s))] or [audio_root]

    def _scan_audio_files(self) -> Dict[str, str]:
        cache = {}
        for d in self.audio_search_dirs:
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
                        if f not in cache: cache[f] = os.path.join(root, f)
        return cache

    def _find_audio_path(self, audio_name: str) -> str:
        if os.path.isabs(audio_name):
            if os.path.exists(audio_name): return audio_name
            raise FileNotFoundError(f"音频不存在: {audio_name}")
        if os.path.dirname(audio_name):
            c = os.path.join(self.task_dir, audio_name)
            if os.path.exists(c): return c
            c = os.path.join(self.task_dir, self.tts_cfg.audio_dir or "audio", audio_name)
            if os.path.exists(c): return c
            raise FileNotFoundError(f"音频不存在: {audio_name}")
        if audio_name in self._audio_path_cache: return self._audio_path_cache[audio_name]
        raise FileNotFoundError(f"音频未找到: {audio_name}")

    def _load_system_prompt(self) -> Optional[str]:
        fp = self.tts_cfg.system_prompt_file
        if not fp: return None
        fp = self._resolve_path(fp, base=self.task_dir)
        if not os.path.exists(fp): return None
        with open(fp, "r", encoding="utf-8") as f: return f.read().strip() or None

    def __len__(self): return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        audio_path = self._find_audio_path(item["audio"])
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        max_in = int(self.tts_cfg.max_audio_length * 16000)
        max_tgt = int(self.tts_cfg.max_target_audio_length * 16000)
        input_wav = waveform[:max_in] if len(waveform) > max_in else waveform
        tgt_wav = waveform[:max_tgt] if len(waveform) > max_tgt else waveform
        sys_text = item.get("system_prompt") or self._global_system_prompt or "请复述用户提供的语音内容。"
        audios, audio_parts = [], []
        msgs = [{"role": "system", "content": sys_text}]
        user_msgs = ["<audio>./</audio>"]
        audios.append(input_wav); audio_parts.append(1)
        user_msgs.append("请复述这段语音。")
        msgs.append({"role": "user", "content": "".join(user_msgs)})
        msgs.append({"role": "assistant", "content": f"<|tts_bos|>{item['text']}<|tts_eos|>"})
        full_prompt = self.processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prefix_msgs = msgs[:-1]
        prefix_prompt = self.processor.tokenizer.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=False)
        return {"prompt": full_prompt, "prefix_prompt": prefix_prompt, "audios": audios, "audio_parts": audio_parts, "target_audio": tgt_wav.copy(), "text": item["text"], "audio_path": audio_path}


# ==================== DataCollator ====================

class TTSDataCollator:
    def __init__(self, processor: MiniCPMOProcessor, device: str, vq_codec: VQCodec, max_length: int = 512):
        self.processor = processor
        self.device = device
        self.vq_codec = vq_codec
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict:
        prompts = [b["prompt"] for b in batch]
        audios_list = [b["audios"] for b in batch]
        audio_parts_list = [b["audio_parts"] for b in batch]
        target_audios = [b["target_audio"] for b in batch]
        audio_paths = [b["audio_path"] for b in batch]

        inputs = self.processor(prompts, [[] for _ in batch], audios_list, audio_parts_list,
                                return_tensors="pt", max_length=self.max_length, padding=True, truncation=True)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        tts_bos_id = self.processor.tokenizer.convert_tokens_to_ids("<|tts_bos|>")
        tts_eos_id = self.processor.tokenizer.convert_tokens_to_ids("<|tts_eos|>")

        tts_bounds = []
        for b in range(input_ids.shape[0]):
            ids = input_ids[b].tolist()
            try:
                bos_idx = ids.index(tts_bos_id)
                eos_cands = [i for i, x in enumerate(ids) if x == tts_eos_id and i > bos_idx]
                eos_idx = eos_cands[0] if eos_cands else -1
                if eos_idx == -1: eos_idx = int(attention_mask[b].sum().item())
            except ValueError:
                bos_idx, eos_idx = -1, -1
            tts_bounds.append((bos_idx, eos_idx))

        target_vq_tokens = [self.vq_codec.encode(a) for a in target_audios]

        labels = input_ids.clone().long()
        pad_id = self.processor.tokenizer.pad_token_id or 0
        for b in range(input_ids.shape[0]):
            ids = input_ids[b].tolist()
            try:
                bos_pos = ids.index(tts_bos_id)
                labels[b, :bos_pos + 1] = -100
            except ValueError:
                labels[b, :] = -100
            try:
                eos_pos = ids.index(tts_eos_id)
                labels[b, eos_pos + 1:] = -100
            except ValueError:
                pass
            labels[b, input_ids[b] == pad_id] = -100

        return {
            "input_ids": input_ids.to(self.device),
            "position_ids": torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0).expand(input_ids.shape[0], -1),
            "attention_mask": attention_mask.to(self.device),
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
            "audio_paths": audio_paths,
            "target_audios": target_audios,
        }


# ==================== 冻结策略 ====================

def apply_freeze_strategy(model: MiniCPMO, cfg: TrainConfig):
    print("\n" + "=" * 60 + "\n应用冻结策略\n" + "=" * 60)
    if hasattr(model, "llm") and model.llm:
        if cfg.freeze_llm:
            for p in model.llm.parameters(): p.requires_grad = False
            if cfg.unfreeze_llm_layers:
                n = len(model.llm.model.layers)
                for idx in cfg.unfreeze_llm_layers:
                    ri = idx if idx >= 0 else n + idx
                    if 0 <= ri < n:
                        for p in model.llm.model.layers[ri].parameters(): p.requires_grad = True
                for p in model.llm.lm_head.parameters(): p.requires_grad = True
                for p in model.llm.model.embed_tokens.parameters(): p.requires_grad = True
                print("  → LLM 部分解冻")
            else:
                print("  → LLM 全部冻结")
        else:
            print("  → LLM 全部可训练")
    if hasattr(model, "vpm") and model.vpm:
        for p in model.vpm.parameters(): p.requires_grad = not cfg.freeze_vision
        for p in model.resampler.parameters(): p.requires_grad = not cfg.freeze_vision
        print(f"  → Vision {'冻结' if cfg.freeze_vision else '可训练'}")
    if hasattr(model, "apm") and model.apm:
        for p in model.apm.parameters(): p.requires_grad = not cfg.freeze_audio
        for p in model.audio_projection_layer.parameters(): p.requires_grad = not cfg.freeze_audio
        print(f"  → Audio {'冻结' if cfg.freeze_audio else '可训练'}")
    if hasattr(model, "tts") and model.tts:
        for p in model.tts.parameters(): p.requires_grad = not cfg.freeze_tts
        print(f"  → TTS {'冻结' if cfg.freeze_tts else '可训练'}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Params] Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.2f}%)")


# ==================== TTS Loss ====================

def compute_tts_loss(model: MiniCPMO, batch: Dict, llm_hidden_states: torch.Tensor):
    device = llm_hidden_states.device
    tts = model.tts
    tts_bounds = batch["tts_bounds"]
    target_vq_tokens = batch["target_vq_tokens"]
    losses, all_pred = [], []

    for b in range(len(tts_bounds)):
        bos_idx, eos_idx = tts_bounds[b]
        if bos_idx < 0 or eos_idx <= bos_idx:
            all_pred.append(None); continue
        sample_ids = batch["input_ids"][b]
        llm_tokens = sample_ids[bos_idx + 1 : eos_idx]
        llm_hidden = llm_hidden_states[b, bos_idx + 1 : eos_idx, :]
        if llm_tokens.numel() == 0 or llm_hidden.shape[0] == 0:
            all_pred.append(None); continue

        llm_embeds = tts.emb_text(llm_tokens)
        proj = tts.projector_semantic(llm_hidden)
        if getattr(tts.config, "normalize_projected_hidden", False):
            proj = F.normalize(proj, p=2, dim=-1)
        tts_embeds = llm_embeds + proj

        spk = torch.zeros(0, tts.config.hidden_size, device=device, dtype=tts_embeds.dtype)
        text_eos_id = getattr(tts.config, "text_eos_token_id", 2)
        audio_bos_id = getattr(tts, "audio_bos_token_id", 1)
        text_eos_emb = tts.emb_text(torch.tensor([text_eos_id], device=device, dtype=torch.long))
        audio_bos_emb = tts.emb_text(torch.tensor([audio_bos_id], device=device, dtype=torch.long))
        condition = torch.cat([spk, tts_embeds, text_eos_emb, audio_bos_emb], dim=0).unsqueeze(0)

        tgt_np = target_vq_tokens[b]
        tgt = torch.from_numpy(tgt_np).long().to(device)
        if tgt.dim() == 1: tgt = tgt.unsqueeze(1)
        if tgt.shape[1] < tts.num_vq: tgt = tgt.repeat(1, tts.num_vq)
        elif tgt.shape[1] > tts.num_vq: tgt = tgt[:, :tts.num_vq]

        seq_len = tgt.shape[0]
        if seq_len == 0: all_pred.append(None); continue

        if seq_len > 1:
            ainp = tgt[:-1, :].unsqueeze(0)
            aemb = []
            for q in range(tts.num_vq):
                aemb.append(tts.emb_code[q](ainp[:, :, q]))
            aemb = torch.stack(aemb, dim=-1).sum(dim=-1)
        else:
            aemb = torch.zeros(1, 0, tts.config.hidden_size, device=device)

        full = torch.cat([condition, aemb], dim=1)
        pos = torch.arange(full.shape[1], device=device).unsqueeze(0)
        out = tts.model(inputs_embeds=full, position_ids=pos, use_cache=False, output_hidden_states=False)
        audio_hid = out.last_hidden_state[:, condition.shape[1] - 1 : condition.shape[1] - 1 + seq_len, :]

        loss_b = 0
        pred_layers = []
        for q in range(tts.num_vq):
            logits_q = tts.head_code[q](audio_hid)
            loss_b += F.cross_entropy(logits_q.reshape(-1, logits_q.size(-1)), tgt[:, q].unsqueeze(0).reshape(-1), ignore_index=-100)
            pred_layers.append(logits_q.argmax(dim=-1).squeeze(0).cpu().numpy())
        losses.append(loss_b / tts.num_vq)
        all_pred.append(pred_layers[0] if pred_layers else None)

    if not losses: return torch.tensor(0.0, device=device), all_pred
    return torch.stack(losses).mean(), all_pred


# ==================== 重建检查器（使用 batch step，不依赖 global_step） ====================
class AudioReconstructionChecker:
    """
    训练时只收集 VQ tokens 中间文件，不执行音频重建（避免 CUDA 报错）。
    离线重建请使用 offline_reconstruct.py。
    """

    def __init__(self, output_dir: str):
        self.output_dir = os.path.join(output_dir, "reconstruct_check")
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger("tts_train")

    def check(
        self,
        step: int,
        pred_tokens: Optional[np.ndarray],
        gt_tokens: np.ndarray,
        target_audio: np.ndarray,
        audio_path: str,
        pred_text: str = "",
        gt_text: str = "",
    ) -> Optional[Dict]:
        if pred_tokens is None or len(pred_tokens) == 0:
            self.logger.warning(f"[Recon] Batch step {step}: pred_tokens 为空，跳过")
            return None

        step_dir = os.path.join(self.output_dir, f"step_{step:06d}")
        os.makedirs(step_dir, exist_ok=True)

        # 保存原始音频（numpy 数组，无需 CUDA）
        sf.write(os.path.join(step_dir, "original_16k.wav"), target_audio, 16000)

        # 保存 VQ token 中间文件
        np.save(os.path.join(step_dir, "pred_tokens.npy"), pred_tokens.astype(np.int64))
        np.save(os.path.join(step_dir, "gt_tokens.npy"), gt_tokens.astype(np.int64))

        # 保存元数据
        meta = {
            "step": step,
            "audio_path": audio_path,
            "pred_text": pred_text,
            "gt_text": gt_text,
            "pred_tokens_shape": list(pred_tokens.shape),
            "pred_tokens_range": [int(pred_tokens.min()), int(pred_tokens.max())],
            "pred_unique_tokens": int(len(np.unique(pred_tokens))),
            "gt_tokens_shape": list(gt_tokens.shape),
            "gt_tokens_range": [int(gt_tokens.min()), int(gt_tokens.max())],
            "gt_unique_tokens": int(len(np.unique(gt_tokens))),
        }
        with open(os.path.join(step_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self.logger.info(
            f"[Recon] Batch step {step}: saved to {step_dir} | "
            f"pred_tokens={pred_tokens.shape} range=[{pred_tokens.min()}, {pred_tokens.max()}] unique={meta['pred_unique_tokens']} | "
            f"gt_tokens={gt_tokens.shape} range=[{gt_tokens.min()}, {gt_tokens.max()}] unique={meta['gt_unique_tokens']}"
        )

        return {
            "step": step,
            "output_dir": step_dir,
            "pred_tokens_path": os.path.join(step_dir, "pred_tokens.npy"),
            "gt_tokens_path": os.path.join(step_dir, "gt_tokens.npy"),
            "meta_path": os.path.join(step_dir, "meta.json"),
        }
    
# ==================== 训练主函数 ====================

def train():
    cfg = load_config("config.yaml")
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger = logging.getLogger("tts_train")
    logger.setLevel(logging.DEBUG if cfg.debug_mode else logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()

    log_file = os.path.join(cfg.output_dir, f"train_{datetime.now().strftime('%m%d_%H%M%S')}.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)

    logger.info(f"Training started. Log file: {log_file}")
    if cfg.debug_no_optimize:
        logger.info(">>> DEBUG MODE: 不执行 optimizer.step()，仅前向 + 重建检查 <<<")
    if cfg.debug_reconstruct:
        logger.info(f"Reconstruction check: every {cfg.debug_reconstruct_steps} batches")

    if cfg.num_workers > 0:
        logger.warning("VQCodec 不可序列化，强制 num_workers=0")
        cfg.num_workers = 0

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, cfg.torch_dtype, torch.bfloat16)

    logger.info("初始化 VQCodec...")
    vq_codec = VQCodec(model_dir=cfg.model_name_or_path, device=str(device), s3tokenizer_name="speech_tokenizer_v2_25hz")

    recon_checker = AudioReconstructionChecker(vq_codec, cfg.output_dir) if cfg.debug_reconstruct else None

    logger.info("加载 MiniCPM-o-4.5...")
    model = MiniCPMO.from_pretrained(cfg.model_name_or_path, trust_remote_code=cfg.trust_remote_code, torch_dtype=dtype).to(device)
    model.config.stream_input = False
    if hasattr(model, "set_mode"):
        model.set_mode({"chat": ProcessorMode.CHAT, "streaming": ProcessorMode.STREAMING, "duplex": ProcessorMode.DUPLEX}.get(cfg.mode, ProcessorMode.CHAT))

    processor = MiniCPMOProcessor.from_pretrained(cfg.model_name_or_path, trust_remote_code=cfg.trust_remote_code)
    bos_id = processor.tokenizer.convert_tokens_to_ids("<|tts_bos|>")
    eos_id = processor.tokenizer.convert_tokens_to_ids("<|tts_eos|>")
    logger.info(f"Tokenizer: <|tts_bos|>={bos_id}, <|tts_eos|>={eos_id}")
    if bos_id == processor.tokenizer.unk_token_id or eos_id == processor.tokenizer.unk_token_id:
        raise RuntimeError("Special token 未正确注册")

    apply_freeze_strategy(model, cfg)

    train_dataset = TTSDataset(cfg.data.tts, processor, cfg, split="train")
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                              collate_fn=TTSDataCollator(processor, str(device), vq_codec, max_length=cfg.max_text_length))

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.num_epochs // cfg.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * cfg.warmup_ratio), num_training_steps=total_steps)

    use_amp = (cfg.device == "cuda" and cfg.torch_dtype == "float16")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    model.train()
    global_step = 0

    for epoch in range(cfg.num_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        epoch_loss = epoch_text_loss = epoch_tts_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(progress_bar):
            # ========== Forward ==========
            with (torch.cuda.amp.autocast(dtype=dtype) if cfg.device == "cuda" else torch.nullcontext()):
                outputs = model(batch, return_dict=True, output_hidden_states=True)
                llm_logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                llm_hidden_states = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else None

                labels = batch["labels"]
                shift_logits = llm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().long()

                valid = (shift_labels != -100).sum()
                if valid > 0:
                    text_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
                else:
                    text_loss = torch.tensor(0.0, device=device)
                    logger.warning(f"[Batch {step}] 所有 labels 为 -100")

                tts_loss = torch.tensor(0.0, device=device)
                pred_tokens_list = []
                if llm_hidden_states is not None and cfg.tts_loss_weight > 0:
                    tts_loss, pred_tokens_list = compute_tts_loss(model, batch, llm_hidden_states)

                loss = text_loss * cfg.text_loss_weight + tts_loss * cfg.tts_loss_weight
                if not cfg.debug_no_optimize:
                    loss = loss / cfg.gradient_accumulation_steps

            # ========== Backward（调试时可跳过） ==========
            if not cfg.debug_no_optimize:
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            # ========== 重建检查（每个 batch 后触发，不依赖 global_step） ==========
            if cfg.debug_reconstruct and recon_checker is not None and step % cfg.debug_reconstruct_steps == 0:
                with torch.no_grad():
                    labels_sample = batch["labels"][0]
                    valid_mask = labels_sample != -100
                    valid_pos = torch.where(valid_mask)[0]
                    pred_text = gt_text = ""
                    if len(valid_pos) > 0:
                        s, e = valid_pos[0].item(), valid_pos[-1].item() + 1
                        gt_text = processor.tokenizer.decode(labels_sample[s:e].cpu().tolist(), skip_special_tokens=False)
                        pred_logits = llm_logits[0, s - 1 : e - 1, :]
                        pred_text = processor.tokenizer.decode(pred_logits.argmax(dim=-1).cpu().tolist(), skip_special_tokens=False)

                    recon_checker.check(
                        step=step,
                        pred_tokens=pred_tokens_list[0] if pred_tokens_list else None,
                        gt_tokens=batch["target_vq_tokens"][0],
                        target_audio=batch["target_audios"][0] if "target_audios" in batch else np.array([]),
                        audio_path=batch["audio_paths"][0] if "audio_paths" in batch else "",
                        pred_text=pred_text[:200],
                        gt_text=gt_text[:200],
                    )

            # 记录
            raw_loss = loss.item() if cfg.debug_no_optimize else loss.item() * cfg.gradient_accumulation_steps
            raw_text = text_loss.item()
            raw_tts = tts_loss.item()
            epoch_loss += raw_loss; epoch_text_loss += raw_text; epoch_tts_loss += raw_tts; num_batches += 1

            progress_bar.set_postfix({
                "loss": f"{raw_loss:.4f}", "text": f"{raw_text:.4f}", "tts": f"{raw_tts:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}" if global_step > 0 else f"{cfg.learning_rate:.2e}",
            })

            # ========== 优化（可选） ==========
            if not cfg.debug_no_optimize and (step + 1) % cfg.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                scheduler.step(); optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.log_steps == 0:
                    logger.info(f"[Global Step {global_step}] text={raw_text:.4f} tts={raw_tts:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

                if global_step % cfg.save_steps == 0:
                    sp = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(sp, exist_ok=True)
                    model.save_pretrained(sp); processor.save_pretrained(sp)
                    logger.info(f"[Save] {sp}")

        logger.info(f"[Epoch {epoch+1}] loss={epoch_loss/num_batches:.4f} text={epoch_text_loss/num_batches:.4f} tts={epoch_tts_loss/num_batches:.4f}")

    if not cfg.debug_no_optimize:
        fp = os.path.join(cfg.output_dir, "final")
        os.makedirs(fp, exist_ok=True)
        model.save_pretrained(fp); processor.save_pretrained(fp)
        logger.info(f"[Done] {fp}")
    else:
        logger.info("[Done] Debug mode (no optimize): 模型权重未保存")

    if cfg.debug_reconstruct and recon_checker:
        logger.info(f"[Recon] 所有重建检查保存在: {recon_checker.output_dir}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    train()
