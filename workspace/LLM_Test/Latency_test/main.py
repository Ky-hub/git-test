#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮对话 Chat 客户端（剧本模式 & 多用户并发）
支持从 config.yaml/config.json 读取配置，流式计算首Token时延(TTFT)
兼容 Qwen 系列模型 /v1/chat/completions 接口

使用方式:
    python chat_client.py                # 默认读取 config.yaml
    python chat_client.py my_config.json # 指定配置文件
    API_URL="https://xxx.com" python chat_client.py

依赖:
    pip install pyyaml requests
"""

import os
import sys
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

# 尝试导入 PyYAML，如未安装则给出友好提示
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

import requests


# ==================== 线程安全输出锁 ====================
_print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    """带锁的线程安全打印，避免多用户并发时输出混乱"""
    with _print_lock:
        print(*args, **kwargs)


# ==================== 配置数据类 ====================

@dataclass
class UserConfig:
    user_id: str
    script_file: str                  # 剧本文件路径（每行一轮用户输入）
    system_prompt_file: str = ""      # 可选：该用户专属的 system prompt 文件
    temperature: float = 0.7
    max_tokens: int = 4096
    think_time: float = 0.0           # 模拟思考间隔（秒）


@dataclass
class GlobalConfig:
    api_url: str
    model: str
    stream: bool = True
    timeout: int = 60
    max_workers: int = 4              # 并发线程数（同时模拟多少用户）
    output_dir: str = "results"
    verbose: bool = True


# ==================== 配置加载 ====================

def load_config(config_path: str) -> tuple[GlobalConfig, List[UserConfig]]:
    """从配置文件加载，支持 .yaml / .yml / .json"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith((".yaml", ".yml")):
            if not HAS_YAML:
                raise ImportError("检测到 YAML 配置文件，但未安装 PyYAML。请执行: pip install pyyaml")
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    # 全局配置（环境变量优先级高于配置文件）
    global_cfg = GlobalConfig(
        api_url=os.getenv("API_URL", data.get("api_url", "https://your-api-endpoint.com")),
        model=os.getenv("MODEL", data.get("model", "qwen3.6-27b")),
        stream=data.get("stream", True),
        timeout=int(os.getenv("TIMEOUT", data.get("timeout", 60))),
        max_workers=data.get("max_workers", 4),
        output_dir=data.get("output_dir", "results"),
        verbose=data.get("verbose", True),
    )

    # 用户配置（程序自动感知 users 列表长度，即并发用户数）
    users = []
    for idx, u in enumerate(data.get("users", [])):
        users.append(UserConfig(
            user_id=u.get("user_id", f"user_{idx+1}"),
            script_file=u["script_file"],
            system_prompt_file=u.get("system_prompt_file", ""),
            temperature=u.get("temperature", 0.7),
            max_tokens=u.get("max_tokens", 4096),
            think_time=u.get("think_time", 0.0),
        ))

    return global_cfg, users


def load_system_prompt(filepath: str) -> str:
    """从外部文件加载 system prompt，文件不存在则返回空字符串"""
    if not filepath:
        return ""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    safe_print(f"[警告] 未找到 system prompt 文件: {filepath}")
    return ""


# ==================== 核心请求逻辑 ====================

def chat_completions(
    messages: list,
    api_url: str,
    model: str,
    stream: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout: int = 60,
):
    """
    调用 /v1/chat/completions 接口（Qwen 系列兼容）
    无需认证（如需要可在 headers 中自行添加 Authorization）
    """
    url = api_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        # 如需要 API Key，取消下面注释并设置环境变量 API_KEY:
        # "Authorization": f"Bearer {os.getenv('API_KEY', '')}"
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            url, headers=headers, json=payload,
            stream=stream, timeout=timeout
        )
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        safe_print(f"\n[请求错误] {e}")
        return None


def parse_stream_with_ttft(response, user_id: str = "", round_idx: int = 0):
    """
    解析 SSE 流式响应，逐字输出并精确计算首Token时延(TTFT)
    返回: (完整回复内容, 首Token时延ms, 总耗时ms)
    """
    full_content = ""
    ttft_ms = 0.0
    total_ms = 0.0
    first_token_received = False
    start_time = time.time()

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue

        # SSE 格式: data: {...}
        if line.startswith("data: "):
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")

                if content:
                    # 首次收到有效内容，记录首Token时延
                    if not first_token_received:
                        first_token_time = time.time()
                        ttft_ms = (first_token_time - start_time) * 1000
                        first_token_received = True
                        safe_print(f"\n[TTFT] {ttft_ms:.2f}ms", end=" ")

                    # 流式输出（线程安全）
                    with _print_lock:
                        print(content, end="", flush=True)
                    full_content += content

            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    total_ms = (time.time() - start_time) * 1000
    print()  # 换行
    return full_content, round(ttft_ms, 2), round(total_ms, 2)


def parse_non_stream_response(response):
    """解析非流式响应（TTFT等于总耗时）"""
    start_time = time.time()
    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        print(content)
        total_ms = (time.time() - start_time) * 1000
        return content, round(total_ms, 2), round(total_ms, 2)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"[解析错误] {e}")
        return "", 0.0, 0.0


# ==================== 剧本执行引擎 ====================

def run_user_script(user: UserConfig, global_cfg: GlobalConfig) -> dict:
    """
    执行单个用户的剧本对话
    每行一个用户输入，自动维护多轮对话历史
    """
    user_id = user.user_id
    # safe_print(f"\n{":"*60}")
    safe_print(f"[用户 {user_id}] 开始执行剧本")
    # safe_print(f"{":"*60}")

    # 加载该用户的 system prompt
    system_prompt = load_system_prompt(user.system_prompt_file)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        safe_print(f"[用户 {user_id}] 已加载 system prompt")

    # 读取剧本文件（跳过空行和 # 注释行）
    if not os.path.exists(user.script_file):
        safe_print(f"[错误] 剧本文件不存在: {user.script_file}")
        return {"user_id": user_id, "error": "剧本文件不存在"}

    with open(user.script_file, "r", encoding="utf-8") as f:
        lines = [
            line.strip() for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if not lines:
        safe_print(f"[警告] 剧本文件为空: {user.script_file}")
        return {"user_id": user_id, "error": "剧本文件为空"}

    safe_print(f"[用户 {user_id}] 共 {len(lines)} 轮对话")

    round_results = []

    for round_idx, user_input in enumerate(lines, 1):
        safe_print(f"\n[用户 {user_id}] 第 {round_idx}/{len(lines)} 轮")
        safe_print(f"User: {user_input}")

        # 模拟真实用户思考间隔
        if user.think_time > 0:
            time.sleep(user.think_time)

        # 追加用户消息到历史
        messages.append({"role": "user", "content": user_input})

        # 发送请求
        print("Assistant: ", end="", flush=True)
        response = chat_completions(
            messages=messages,
            api_url=global_cfg.api_url,
            model=global_cfg.model,
            stream=global_cfg.stream,
            temperature=user.temperature,
            max_tokens=user.max_tokens,
            timeout=global_cfg.timeout,
        )

        if response is None:
            safe_print(f"[用户 {user_id}] 请求失败，移除本轮用户消息")
            messages.pop()
            round_results.append({
                "round": round_idx,
                "user_input": user_input,
                "error": "请求失败",
                "ttft_ms": 0,
                "total_ms": 0,
                "timestamp": datetime.now().isoformat(),
            })
            continue

        # 解析响应并计算时延
        if global_cfg.stream:
            assistant_content, ttft_ms, total_ms = parse_stream_with_ttft(
                response, user_id=user_id, round_idx=round_idx
            )
        else:
            assistant_content, ttft_ms, total_ms = parse_non_stream_response(response)

        # 记录本轮结果
        round_result = {
            "round": round_idx,
            "user_input": user_input,
            "assistant_output": assistant_content,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "timestamp": datetime.now().isoformat(),
        }
        round_results.append(round_result)

        # 将助手回复加入历史，维持多轮对话上下文
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})

        safe_print(f"[用户 {user_id}] 第 {round_idx} 轮完成 | "
                   f"TTFT: {ttft_ms}ms | Total: {total_ms}ms")

    # 汇总统计
    valid_rounds = [r for r in round_results if "error" not in r]
    summary = {
        "user_id": user_id,
        "total_rounds": len(lines),
        "completed_rounds": len(valid_rounds),
        "avg_ttft_ms": round(
            sum(r["ttft_ms"] for r in valid_rounds) / len(valid_rounds), 2
        ) if valid_rounds else 0,
        "avg_total_ms": round(
            sum(r["total_ms"] for r in valid_rounds) / len(valid_rounds), 2
        ) if valid_rounds else 0,
        "details": round_results,
    }

    safe_print(f"\n[用户 {user_id}] 剧本执行完成")
    safe_print(f"  完成轮数: {summary['completed_rounds']}/{summary['total_rounds']}")
    safe_print(f"  平均首Token时延: {summary['avg_ttft_ms']} ms")
    safe_print(f"  平均总耗时: {summary['avg_total_ms']} ms")

    return summary


# ==================== 结果持久化 ====================

def save_results(results: List[dict], output_dir: str):
    """保存测试结果到 JSON（详细）和 CSV（摘要）"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 保存详细 JSON
    json_path = os.path.join(output_dir, f"benchmark_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 2. 保存 CSV（方便 Excel 分析）
    csv_path = os.path.join(output_dir, f"benchmark_{timestamp}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("user_id,round,user_input,ttft_ms,total_ms,timestamp\n")
        for user_res in results:
            user_id = user_res.get("user_id", "unknown")
            for detail in user_res.get("details", []):
                # 对 CSV 中的逗号和换行进行简单转义
                inp = str(detail.get("user_input", "")).replace('"', '""').replace("\n", " ")
                f.write(f"{user_id},{detail['round']},\"{inp}\","
                        f"{detail['ttft_ms']},{detail['total_ms']},"
                        f"{detail.get('timestamp', '')}\n")

    safe_print(f"\n[结果已保存]")
    safe_print(f"  详细报告: {json_path}")
    safe_print(f"  CSV 报表: {csv_path}")
    return json_path, csv_path


# ==================== 主程序 ====================

def main():
    # 配置文件路径：默认 config.yaml，可通过命令行参数或环境变量指定
    config_path = os.getenv("CONFIG_FILE", "config.yaml")
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    # 如果配置文件不存在，生成示例并退出
    if not os.path.exists(config_path):
        safe_print(f"[错误] 配置文件不存在: {config_path}")
        safe_print("\n请创建配置文件，例如 config.yaml：")
        safe_print("""
# ========== config.yaml 示例 ==========
api_url: "https://your-api-endpoint.com"   # Qwen API 地址
model: "qwen3.6-27b"                       # 模型名称
stream: true                               # 必须 true 才能测 TTFT
timeout: 60
max_workers: 3                             # 并发用户数
output_dir: "results"
verbose: true

users:
  - user_id: "user_1"
    script_file: "script_user1.txt"
    system_prompt_file: "system_prompt_1.txt"
    temperature: 0.7
    max_tokens: 4096
    think_time: 1.0                        # 每轮间隔 1 秒

  - user_id: "user_2"
    script_file: "script_user2.txt"
    system_prompt_file: "system_prompt_2.txt"
    temperature: 0.5
    max_tokens: 2048
    think_time: 0.5

  - user_id: "user_3"
    script_file: "script_user3.txt"
    # 不配置 system_prompt_file 则表示不使用
    temperature: 0.7
    max_tokens: 4096
    think_time: 2.0
""")
        sys.exit(1)

    # 加载配置
    try:
        global_cfg, users = load_config(config_path)
    except Exception as e:
        safe_print(f"[配置错误] {e}")
        sys.exit(1)

    if not users:
        safe_print("[错误] 配置文件中未定义任何用户")
        sys.exit(1)

    # 检查 API 端点
    if global_cfg.api_url.endswith("your-api-endpoint.com"):
        safe_print("[错误] 请配置正确的 api_url（环境变量 API_URL 或配置文件）")
        sys.exit(1)

    # safe_print(f"\n{":"*60}")
    safe_print("[系统初始化]")
    safe_print(f"  API 端点: {global_cfg.api_url}")
    safe_print(f"  模型: {global_cfg.model}")
    safe_print(f"  流式模式: {global_cfg.stream}")
    safe_print(f"  并发用户数: {len(users)} (max_workers={global_cfg.max_workers})")
    safe_print(f"  输出目录: {global_cfg.output_dir}")
    # safe_print(f"{":"*60}")

    # 执行多用户并发测试
    all_results = []

    if len(users) == 1:
        # 单用户：直接执行，无需线程池
        result = run_user_script(users[0], global_cfg)
        all_results.append(result)
    else:
        # 多用户：使用线程池并发执行
        with ThreadPoolExecutor(max_workers=global_cfg.max_workers) as executor:
            future_to_user = {
                executor.submit(run_user_script, user, global_cfg): user
                for user in users
            }

            for future in as_completed(future_to_user):
                user = future_to_user[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    safe_print(f"[用户 {user.user_id}] 执行异常: {e}")
                    all_results.append({
                        "user_id": user.user_id,
                        "error": str(e)
                    })

    # 保存结果
    save_results(all_results, global_cfg.output_dir)

    # 最终汇总
    # safe_print(f"\n{":"*60}")
    safe_print("全部测试完成 - 总摘要")
    # safe_print(f"{":"*60}")
    for r in all_results:
        uid = r.get("user_id", "unknown")
        if "error" in r:
            safe_print(f"[{uid}] 异常: {r['error']}")
        else:
            safe_print(f"[{uid}] 完成: {r['completed_rounds']}/{r['total_rounds']} 轮 | "
                       f"平均TTFT: {r['avg_ttft_ms']} ms | 平均Total: {r['avg_total_ms']} ms")


if __name__ == "__main__":
    main()
