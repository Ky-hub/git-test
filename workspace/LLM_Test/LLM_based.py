#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮对话 Chat 客户端（无需 API Key）
支持从外部文件读取 system_prompt，使用 requests 库调用 /v1/chat/completions
"""

import os
import sys
import json
import requests


# ==================== 配置区域 ====================
API_URL = os.getenv("API_URL", "https://your-api-endpoint.com")
MODEL = os.getenv("MODEL", "qwen3.6-27b")

# system_prompt 文件路径
SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE", "system_prompt.txt")

# 是否开启流式输出
STREAM = os.getenv("STREAM", "true").lower() in ("true", "1", "yes")

# 请求超时时间（秒）
TIMEOUT = int(os.getenv("TIMEOUT", "60"))
# =================================================


def load_system_prompt(filepath: str) -> str:
    """从外部文件加载 system prompt，文件不存在则返回空字符串"""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    print(f"[警告] 未找到 system prompt 文件: {filepath}，将不使用 system prompt")
    return ""


def chat_completions(
    messages: list,
    api_url: str,
    model: str,
    stream: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 4096,
):
    """
    调用 /v1/chat/completions 接口（无需认证）
    """
    url = api_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, stream=stream, timeout=TIMEOUT)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"\n[请求错误] {e}")
        return None


def parse_stream_response(response):
    """解析流式响应，逐字输出并返回完整内容"""
    full_content = ""
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        # 处理 SSE 格式: data: {...}
        if line.startswith("data: "):
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
                    full_content += content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    print()  # 换行
    return full_content


def parse_non_stream_response(response):
    """解析非流式响应"""
    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        print(content)
        return content
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"[解析错误] 无法解析响应: {e}")
        return ""


def main():
    # 检查必要配置
    if API_URL.endswith("your-api-endpoint.com"):
        print("[错误] 请设置环境变量 API_URL")
        print("示例:")
        print('  export API_URL="https://api.example.com"')
        sys.exit(1)

    # 加载 system prompt
    system_prompt = load_system_prompt(SYSTEM_PROMPT_FILE)

    # 初始化对话历史
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        print(f"[已加载 system prompt] 来源: {SYSTEM_PROMPT_FILE}")
    else:
        print("[提示] 未使用 system prompt")

    print(f"[模型] {MODEL}")
    print(f"[接口] {API_URL}/v1/chat/completions")
    print(f"[模式] {'流式' if STREAM else '非流式'}")
    print("-" * 50)
    print("开始对话，输入 'exit' 或 'quit' 退出，输入 'clear' 清空历史\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[退出]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("[退出]")
            break

        if user_input.lower() == "clear":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            print("[历史已清空]")
            continue

        # 添加用户消息
        messages.append({"role": "user", "content": user_input})

        # 发送请求
        print("Assistant: ", end="", flush=True)
        response = chat_completions(messages, API_URL, MODEL, stream=STREAM)

        if response is None:
            # 请求失败，移除刚添加的用户消息，避免污染历史
            messages.pop()
            continue

        if STREAM:
            assistant_content = parse_stream_response(response)
        else:
            assistant_content = parse_non_stream_response(response)

        # 将助手回复加入历史，维持多轮对话
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})

        print()


if __name__ == "__main__":
    main()
