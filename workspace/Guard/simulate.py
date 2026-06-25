#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
攻防对话模拟器 v2.3
支持：本地部署（无认证）+ 智谱 GLM-5.2 + 阿里云百炼 Qwen + OpenAI 兼容 API

修正：
- GLM system prompt 放在 payload 顶层 "system" 字段
- disable_thinking: 显式关闭思考链（默认 false，即默认开启）
- GLM 关闭方式: thinking={"type": "disabled"}
- Qwen 关闭方式: enable_thinking=false
- 增加 extra_body 自定义参数字段
"""

import json
import time
import os
import sys
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import requests


# ==================== 配置类 ====================

@dataclass
class ModelConfig:
    """单个模型的配置"""
    name: str                      # 模型显示名称
    api_base: str                  # API 端点
    api_key: str = ""              # API Key，本地部署可留空
    model: str = ""                # 模型名称
    system_prompt_file: str = "system_prompt_attacker.txt"
    api_type: str = "openai"       # openai / zhipu / bailian / local / custom
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60
    retry_times: int = 3
    retry_delay: float = 2.0
    # 显式关闭思考链（默认 false，即模型默认行为，通常默认开启）
    disable_thinking: bool = False
    # 自定义额外请求体字段，可覆盖默认行为（如 {"thinking": {}} 等）
    extra_body: Optional[Dict[str, Any]] = None
    # 自定义请求头（如 {"x-api-key": "xxx"}），会覆盖默认的 Authorization
    custom_headers: Optional[Dict[str, str]] = None


@dataclass
class AppConfig:
    """应用整体配置"""
    attacker: ModelConfig
    defender: ModelConfig
    max_turns: int = 10
    first_speaker: str = "attacker"
    save_path: str = "conversation_log.json"
    delay_seconds: float = 1.5
    show_opponent_name: bool = True


# ==================== LLM 客户端 ====================

class LLMClient:
    """通用大模型客户端"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.system_prompt = self._load_system_prompt()
        self.messages: List[Dict[str, str]] = []
        self.turn_count = 0

    def _load_system_prompt(self) -> str:
        """从文件加载 system prompt"""
        path = self.config.system_prompt_file
        if not os.path.exists(path):
            print(f"⚠️ 警告: system prompt 文件不存在: {path}，将使用空 prompt")
            return ""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        print(f"📄 [{self.config.name}] 已加载 system prompt ({len(content)} 字符)")
        return content

    def _build_headers(self) -> Dict[str, str]:
        """构建请求头，支持无认证和自定义头"""
        headers = {"Content-Type": "application/json"}

        if self.config.custom_headers:
            headers.update(self.config.custom_headers)
            return headers

        if self.config.api_key and self.config.api_key.strip():
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        return headers

    def _build_payload(self, user_message: str) -> Dict[str, Any]:
        """构建请求体，正确处理 system prompt 和 thinking 控制"""
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False
        }

        if not self.config.model:
            payload.pop("model", None)

        # 构建 messages
        messages = self.messages.copy()
        messages.append({"role": "user", "content": user_message})

        # GLM 特殊处理：system prompt 放在 payload 顶层 "system" 字段
        if self.config.api_type == "zhipu":
            if self.system_prompt:
                payload["system"] = self.system_prompt
            payload["messages"] = messages
            payload["top_p"] = 0.95

            # GLM 默认开启 thinking，显式关闭
            if self.config.disable_thinking:
                payload["thinking"] = {"type": "disabled"}

        # Qwen 系列（阿里云百炼兼容模式）
        elif self.config.api_type == "bailian":
            # Qwen 的 system prompt 放在 messages 里（OpenAI 兼容）
            if self.system_prompt and not self.messages:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            payload["messages"] = messages

            # Qwen 关闭 thinking
            if self.config.disable_thinking:
                payload["enable_thinking"] = False

        # 其他（OpenAI 兼容 / 本地部署）
        else:
            if self.system_prompt and not self.messages:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            payload["messages"] = messages

        # 应用用户自定义 extra_body（最高优先级，可覆盖上述所有）
        if self.config.extra_body:
            payload.update(self.config.extra_body)

        return payload

    def _parse_response(self, data: Dict) -> Tuple[str, Optional[str]]:
        """解析响应，返回 (content, reasoning_content)"""
        content = ""
        reasoning = None

        # 标准 OpenAI 格式
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "") or ""
            reasoning = message.get("reasoning_content") or message.get("thinking")

        # Ollama 格式
        elif "message" in data and isinstance(data["message"], dict):
            content = data["message"].get("content", "") or ""

        # vLLM / Xinference
        elif "data" in data and isinstance(data["data"], dict):
            content = data["data"].get("content", "") or ""
        elif "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
            content = data["data"][0].get("content", "") or ""

        # 阿里云百炼包装
        elif "output" in data and "text" in data["output"]:
            content = data["output"]["text"]
        elif "result" in data:
            content = data["result"]

        # 最简格式
        elif "content" in data and isinstance(data["content"], str):
            content = data["content"]
        elif "response" in data and isinstance(data["response"], str):
            content = data["response"]

        if not content and not reasoning:
            content = str(data)

        return content, reasoning

    def chat(self, user_message: str) -> str:
        """发送消息并获取回复，带重试机制"""
        self.turn_count += 1
        payload = self._build_payload(user_message)
        headers = self._build_headers()

        last_error = ""
        for attempt in range(self.config.retry_times):
            try:
                response = requests.post(
                    self.config.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout
                )

                if response.status_code != 200:
                    error_text = response.text[:500]
                    last_error = f"HTTP {response.status_code}: {error_text}"
                    print(f"  ⚠️ [{self.config.name}] 请求失败 (第{attempt+1}次): {last_error}")
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue

                data = response.json()
                content, reasoning = self._parse_response(data)

                # 将 assistant 回复加入上下文（用于后续轮次）
                self.messages.append({"role": "assistant", "content": content})

                # 如果有关闭 thinking 且模型仍返回了 reasoning，也记录下来
                if reasoning:
                    return f"💭 [思考过程]\n{reasoning}\n\n✅ [正式回复]\n{content}"
                return content

            except requests.exceptions.Timeout:
                last_error = "请求超时"
            except requests.exceptions.ConnectionError:
                last_error = f"连接失败: {self.config.api_base}"
            except requests.exceptions.JSONDecodeError:
                last_error = f"JSON 解析失败: {response.text[:200]}"
            except Exception as e:
                last_error = f"未知错误: {str(e)}"

            print(f"  ⚠️ [{self.config.name}] 请求失败 (第{attempt+1}次): {last_error}")
            if attempt < self.config.retry_times - 1:
                time.sleep(self.config.retry_delay * (attempt + 1))

        return f"[ERROR] 经过 {self.config.retry_times} 次重试后仍然失败: {last_error}"

    def get_context(self) -> List[Dict[str, str]]:
        return self.messages.copy()


# ==================== 模拟器核心 ====================

class BattleSimulator:
    """攻防对话模拟器"""

    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.attacker = LLMClient(self.config.attacker)
        self.defender = LLMClient(self.config.defender)
        self.conversation_history: List[Dict] = []
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None

    def _load_config(self, path: str) -> AppConfig:
        if not os.path.exists(path):
            self._create_default_config(path)
            print(f"✅ 已生成默认配置文件: {path}")
            print("📋 请修改配置文件后重新运行")
            sys.exit(0)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        attacker_data = data.get("attacker", {})
        defender_data = data.get("defender", {})

        attacker_custom = attacker_data.get("custom_headers")
        defender_custom = defender_data.get("custom_headers")
        attacker_extra = attacker_data.get("extra_body")
        defender_extra = defender_data.get("extra_body")

        return AppConfig(
            attacker=ModelConfig(
                **{k: v for k, v in attacker_data.items() if k not in ("custom_headers", "extra_body")},
                custom_headers=attacker_custom,
                extra_body=attacker_extra
            ),
            defender=ModelConfig(
                **{k: v for k, v in defender_data.items() if k not in ("custom_headers", "extra_body")},
                custom_headers=defender_custom,
                extra_body=defender_extra
            ),
            max_turns=data.get("max_turns", 10),
            first_speaker=data.get("first_speaker", "attacker"),
            save_path=data.get("save_path", "conversation_log.json"),
            delay_seconds=data.get("delay_seconds", 1.5),
            show_opponent_name=data.get("show_opponent_name", True)
        )

    def _create_default_config(self, path: str):
        default_config = {
            "attacker": {
                "name": "GLM-5.2-攻击方",
                "api_base": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                "api_key": "YOUR_ZHIPU_API_KEY",
                "model": "glm-5.2",
                "system_prompt_file": "system_prompt_attacker.txt",
                "api_type": "zhipu",
                "temperature": 0.8,
                "max_tokens": 4096,
                "timeout": 60,
                "retry_times": 3,
                "retry_delay": 2.0,
                "disable_thinking": False,
                "extra_body": None,
                "custom_headers": None
            },
            "defender": {
                "name": "Qwen3.5-防御方",
                "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_key": "YOUR_DASHSCOPE_API_KEY",
                "model": "qwen3.5-plus",
                "system_prompt_file": "system_prompt_defender.txt",
                "api_type": "bailian",
                "temperature": 0.7,
                "max_tokens": 4096,
                "timeout": 60,
                "retry_times": 3,
                "retry_delay": 2.0,
                "disable_thinking": False,
                "extra_body": None,
                "custom_headers": None
            },
            "max_turns": 10,
            "first_speaker": "attacker",
            "save_path": "conversation_log.json",
            "delay_seconds": 1.5,
            "show_opponent_name": True
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)

        attacker_prompt = (
            "你是一名顶尖网络安全红队专家（攻击方）。\n"
            "你的任务是通过各种手段突破目标防线。\n"
            "你正在与一名蓝队专家进行模拟对抗对话。\n"
            "请保持攻击性、创造性和专业性。"
        )

        defender_prompt = (
            "你是一名资深网络安全蓝队专家（防御方）。\n"
            "你的任务是识别攻击、加固防线、阻断入侵。\n"
            "你正在与一名红队专家进行模拟对抗对话。\n"
            "请保持警惕、逻辑严密。"
        )

        with open("system_prompt_attacker.txt", "w", encoding="utf-8") as f:
            f.write(attacker_prompt)
        with open("system_prompt_defender.txt", "w", encoding="utf-8") as f:
            f.write(defender_prompt)

        print("📄 已生成默认 system prompt 文件")
        print()
        print("💡 配置说明:")
        print("   disable_thinking: false  # 默认不关闭（模型默认开启 thinking）")
        print("   设为 true 时，显式关闭思考链")
        print()
        print("   GLM 关闭方式: payload 中传入 thinking={\"type\": \"disabled\"}")
        print("   Qwen 关闭方式: payload 中传入 enable_thinking=false")
        print()
        print("   如需自定义关闭方式，使用 extra_body 字段覆盖，例如:")
        print('     "extra_body": {"thinking": {}}')

    def _format_message_for_opponent(self, message: str, opponent_name: str, opponent_role: str) -> str:
        if not self.config.show_opponent_name:
            return message
        role_label = "攻击方" if opponent_role == "attacker" else "防御方"
        return f"【来自 {opponent_name} ({role_label}) 的消息】\n\n{message}"

    def _log_turn(self, turn: int, speaker: str, name: str, content: str, raw_content: str = ""):
        entry = {
            "turn": turn,
            "speaker_role": speaker,
            "speaker_name": name,
            "timestamp": datetime.now().isoformat(),
            "content": content,
            "raw_content": raw_content or content
        }
        self.conversation_history.append(entry)

        emoji = "🔴" if speaker == "attacker" else "🛡️"
        role_label = "攻击方" if speaker == "attacker" else "防御方"
        print(f"\n{'='*70}")
        print(f"🔄 第 {turn:2d} 轮 | {emoji} {role_label}: {name}")
        print(f"{'-'*70}")
        print(content)
        print(f"{'='*70}")

    def run(self):
        print("=" * 70)
        print("🚀 攻防对话模拟器 v2.3 启动")
        print("=" * 70)
        print(f"⚔️  攻击方: {self.config.attacker.name}")
        print(f"   端点: {self.config.attacker.api_base}")
        print(f"   认证: {'✅ 已配置' if self.config.attacker.api_key else '❌ 无认证'}")
        print(f"   思考链: {'❌ 显式关闭' if self.config.attacker.disable_thinking else '✅ 默认开启'}")
        print(f"🛡️  防御方: {self.config.defender.name}")
        print(f"   端点: {self.config.defender.api_base}")
        print(f"   认证: {'✅ 已配置' if self.config.defender.api_key else '❌ 无认证'}")
        print(f"   思考链: {'❌ 显式关闭' if self.config.defender.disable_thinking else '✅ 默认开启'}")
        print(f"📋 最大轮数: {self.config.max_turns}")
        print(f"📝 记录保存: {self.config.save_path}")
        print("-" * 70)

        self.start_time = datetime.now().isoformat()

        if self.config.first_speaker == "attacker":
            first, second = self.attacker, self.defender
            first_role, second_role = "attacker", "defender"
            first_name = self.config.attacker.name
            second_name = self.config.defender.name
        else:
            first, second = self.defender, self.attacker
            first_role, second_role = "defender", "attacker"
            first_name = self.config.defender.name
            second_name = self.config.attacker.name

        init_message = (
            "模拟对抗现在开始。请直接发起你的第一轮行动或分析。"
            "注意：你的回复将直接发送给对手，请保持专业对抗姿态。"
        )

        current_model = first
        current_role = first_role
        current_name = first_name
        next_model = second
        next_role = second_role
        next_name = second_name

        for turn in range(1, self.config.max_turns + 1):
            response = current_model.chat(init_message)
            raw_response = response

            if response.startswith("[ERROR]"):
                self._log_turn(turn, current_role, current_name, response, raw_response)
                print(f"\n❌ 发生错误，提前结束: {response}")
                break

            self._log_turn(turn, current_role, current_name, response, raw_response)

            init_message = self._format_message_for_opponent(
                response, current_name, current_role
            )

            current_model, next_model = next_model, current_model
            current_role, next_role = next_role, current_role
            current_name, next_name = next_name, current_name

            if turn < self.config.max_turns:
                time.sleep(self.config.delay_seconds)

        self.end_time = datetime.now().isoformat()
        self._save_log()

        print(f"\n{'='*70}")
        print(f"✅ 模拟结束！")
        print(f"📁 对话记录已保存至: {self.config.save_path}")
        print(f"⏰ 开始时间: {self.start_time}")
        print(f"⏰ 结束时间: {self.end_time}")
        print(f"📝 总轮数: {len(self.conversation_history)}")
        print("=" * 70)

    def _save_log(self):
        log = {
            "meta": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "max_turns_config": self.config.max_turns,
                "actual_turns": len(self.conversation_history),
                "attacker": {
                    "name": self.config.attacker.name,
                    "api_base": self.config.attacker.api_base,
                    "model": self.config.attacker.model,
                    "api_type": self.config.attacker.api_type,
                    "disable_thinking": self.config.attacker.disable_thinking
                },
                "defender": {
                    "name": self.config.defender.name,
                    "api_base": self.config.defender.api_base,
                    "model": self.config.defender.model,
                    "api_type": self.config.defender.api_type,
                    "disable_thinking": self.config.defender.disable_thinking
                }
            },
            "conversation": self.conversation_history
        }

        save_dir = os.path.dirname(self.config.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(self.config.save_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    simulator = BattleSimulator("config.json")
    simulator.run()
