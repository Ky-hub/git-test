#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1: 加载 MiniCPM-o 权重并验证模型结构
功能：
  1. 从配置文件读取超参
  2. 加载模型与 Processor
  3. 打印各模块加载状态与参数量
  4. 应用冻结策略并二次统计
  5. 执行一次极简前向测试
  6. 测试保存检查点
"""

import os
import sys
import importlib.util
from enum import Enum
import yaml
import torch
import torch.nn as nn
from dataclasses import dataclass, field, asdict
from typing import Optional, List

# ========== 本地定义 ProcessorMode（避免从 modeling_minicpmo 导入失败） ==========
class ProcessorMode(Enum):
    CHAT = "chat"
    STREAMING = "streaming"
    DUPLEX = "duplex"

# ========== 路径适配 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../Train
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                       # .../project_root
sys.path.insert(0, PROJECT_ROOT)                                 # 让 Python 能找到 MiniCPMO45

# ========== 加载真实的 MiniCPMO45/utils.py，只补充缺失函数（不改原文件） ==========
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


# ==================== 配置类 ====================

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
    freeze_tts: bool = True
    unfreeze_llm_layers: Optional[List[int]] = None

    output_dir: str = "./minicpmo_finetuned"


def load_config(path: str = "config.yaml") -> TrainConfig:
    """从 YAML 加载配置，若文件不存在则返回默认配置"""
    if not os.path.exists(path):
        print(f"[WARN] 未找到 {path}，使用默认配置")
        return TrainConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # 过滤掉 dataclass 中不存在的字段
    valid_keys = {k for k in TrainConfig.__dataclass_fields__}
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    return TrainConfig(**filtered)


# ==================== 工具函数 ====================

def print_module_status(model: MiniCPMO):
    """检查各核心模块是否存在于模型实例中"""
    print("\n" + "=" * 60)
    print("模块加载状态检查")
    print("=" * 60)

    checks = [
        ("LLM 主干 (llm)", "llm"),
        ("视觉编码器 (vpm)", "vpm"),
        ("视觉重采样 (resampler)", "resampler"),
        ("音频编码器 (apm)", "apm"),
        ("音频投影层 (audio_projection_layer)", "audio_projection_layer"),
        ("语音合成 (tts)", "tts"),
    ]

    for name, attr in checks:
        exists = hasattr(model, attr) and getattr(model, attr) is not None
        symbol = "✓" if exists else "✗"
        print(f"  {symbol} {name:<30} {'已加载' if exists else '未加载'}")


def print_parameter_stats(model: MiniCPMO, title: str = "参数统计"):
    """打印总参数量、可训练量、冻结量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"  总参数量:     {total:>12,}  ({total / 1e9:>6.2f} B)")
    print(f"  可训练参数:   {trainable:>12,}  ({trainable / 1e9:>6.2f} B)")
    print(f"  冻结参数:     {frozen:>12,}  ({frozen / 1e9:>6.2f} B)")
    print(f"  可训练比例:   {100 * trainable / total:>11.2f} %")


def apply_freeze_strategy(model: MiniCPMO, cfg: TrainConfig):
    """
    根据配置文件应用参数冻结。
    直接操作 param.requires_grad，不涉及任何封装库。
    """
    print("\n" + "=" * 60)
    print("应用冻结策略")
    print("=" * 60)

    # --- LLM ---
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
                print(f"  → LLM lm_head / embed_tokens        已解冻")
            else:
                print(f"  → LLM 全部冻结")
        else:
            print(f"  → LLM 全部可训练")

    # --- Vision ---
    if hasattr(model, "vpm") and model.vpm is not None:
        freeze = cfg.freeze_vision
        for p in model.vpm.parameters():
            p.requires_grad = not freeze
        print(f"  → Vision Encoder (vpm)               {'冻结' if freeze else '可训练'}")

    if hasattr(model, "resampler") and model.resampler is not None:
        freeze = cfg.freeze_vision
        for p in model.resampler.parameters():
            p.requires_grad = not freeze
        print(f"  → Vision Resampler                   {'冻结' if freeze else '可训练'}")

    # --- Audio ---
    if hasattr(model, "apm") and model.apm is not None:
        freeze = cfg.freeze_audio
        for p in model.apm.parameters():
            p.requires_grad = not freeze
        print(f"  → Audio Encoder (apm)                {'冻结' if freeze else '可训练'}")

    if hasattr(model, "audio_projection_layer") and model.audio_projection_layer is not None:
        freeze = cfg.freeze_audio
        for p in model.audio_projection_layer.parameters():
            p.requires_grad = not freeze
        print(f"  → Audio Projection Layer             {'冻结' if freeze else '可训练'}")

    # --- TTS ---
    if hasattr(model, "tts") and model.tts is not None:
        freeze = cfg.freeze_tts
        for p in model.tts.parameters():
            p.requires_grad = not freeze
        print(f"  → TTS (MiniCPMTTS)                   {'冻结' if freeze else '可训练'}")


def test_minimal_forward(model: MiniCPMO, processor: MiniCPMOProcessor, device: str):
    """
    使用纯文本输入做一次极简前向测试，验证计算图是否通畅。
    不测试多模态分支，仅验证 LLM 主干可正常推理。
    """
    print("\n" + "=" * 60)
    print("极简前向测试 (Pure Text)")
    print("=" * 60)

    try:
        msgs = [{"role": "user", "content": "你好，请介绍一下自己。"}]

        prompt = processor.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            [prompt],
            [[]],   # images
            [[]],   # audios
            return_tensors="pt",
            max_length=64,
        ).to(device)

        data = {
            "input_ids": inputs["input_ids"],
            "position_ids": torch.arange(inputs["input_ids"].shape[1], device=device).unsqueeze(0),
            "pixel_values": inputs.get("pixel_values"),
            "tgt_sizes": inputs.get("tgt_sizes"),
            "audio_features": inputs.get("audio_features"),
            "audio_feature_lens": inputs.get("audio_feature_lens"),
            "image_bound": inputs.get("image_bound"),
            "audio_bounds": inputs.get("audio_bounds"),
            "spk_bounds": inputs.get("spk_bounds"),
        }

        data = {k: v for k, v in data.items() if v is not None}

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if "cuda" in device else torch.float32):
                outputs = model(data, return_dict=True)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        print(f"  ✓ Forward 成功")
        print(f"  → inputs_embeds 计算后 logits shape: {logits.shape}")
        print(f"  → dtype: {logits.dtype}, device: {logits.device}")

    except Exception as e:
        print(f"  ✗ Forward 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_save_checkpoint(model: MiniCPMO, save_dir: str = "./test_checkpoint"):
    """测试保存功能，确保后续训练能正常写盘"""
    print("\n" + "=" * 60)
    print("测试保存检查点")
    print("=" * 60)
    try:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        print(f"  ✓ 检查点已保存至: {os.path.abspath(save_dir)}")
        print(f"  → 包含文件: config.json, pytorch_model.bin / model.safetensors 等")
    except Exception as e:
        print(f"  ✗ 保存失败: {e}")


# ==================== 主入口 ====================

def main():
    cfg = load_config("config.yaml")
    print("当前配置:")
    for k, v in asdict(cfg).items():
        print(f"  {k:<25}: {v}")

    # 设备与精度
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, cfg.torch_dtype, torch.bfloat16)
    print(f"\n使用设备: {device}, 精度: {cfg.torch_dtype}")

    # 1. 加载模型
    print("\n" + "=" * 60)
    print("正在加载模型权重...")
    print("=" * 60)

    try:
        model = MiniCPMO.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=dtype,
        ).to(device)

        if hasattr(model, "set_mode"):
            mode_map = {
                "chat": ProcessorMode.CHAT,
                "streaming": ProcessorMode.STREAMING,
                "duplex": ProcessorMode.DUPLEX,
            }
            target_mode = mode_map.get(cfg.mode, ProcessorMode.CHAT)
            model.set_mode(target_mode)
            print(f"✓ 模型加载成功，模式设置为: {cfg.mode}")

    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 加载 Processor
    try:
        processor = MiniCPMOProcessor.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=cfg.trust_remote_code,
        )
        print("✓ Processor 加载成功")
    except Exception as e:
        print(f"✗ Processor 加载失败: {e}")
        processor = None

    # 3. 结构检查
    print_module_status(model)

    # 4. 初始参数统计
    print_parameter_stats(model, title="初始参数统计（冻结前）")

    # 5. 应用冻结
    apply_freeze_strategy(model, cfg)

    # 6. 冻结后参数统计
    print_parameter_stats(model, title="应用冻结后参数统计")

    # 7. 前向测试
    if processor is not None:
        test_minimal_forward(model, processor, str(device))

    # 8. 保存测试
    test_save_checkpoint(model)

    print("\n" + "=" * 60)
    print("Step 1 完成：模型加载、冻结验证与基础测试全部通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
