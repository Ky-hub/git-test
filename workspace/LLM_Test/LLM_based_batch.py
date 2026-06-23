#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量单轮对话客户端（无需 API Key）
从 example_questions.txt 读取问题，逐条调用 /v1/chat/completions，结果保存到文件
"""

import os
import sys
import json
import time
from datetime import datetime
import requests


# ==================== 配置区域 ====================
API_URL = os.getenv("API_URL", "https://your-api-endpoint.com")
MODEL = os.getenv("MODEL", "qwen3.6-27b")

# system_prompt 文件路径
SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE", "system_prompt.txt")

# 问题列表文件路径
QUESTIONS_FILE = os.getenv("QUESTIONS_FILE", "example_questions.txt")

# 结果保存路径（支持 .json 或 .txt）
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "results.json")

# 请求间隔（秒），防止请求过快
DELAY = float(os.getenv("DELAY", "0.5"))

# 请求超时时间（秒）
TIMEOUT = int(os.getenv("TIMEOUT", "120"))

# 温度与最大 token
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
# =================================================


def load_system_prompt(filepath: str) -> str:
    """从外部文件加载 system prompt"""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    print(f"[警告] 未找到 system prompt 文件: {filepath}，将不使用 system prompt")
    return ""


def load_questions(filepath: str) -> list:
    """从文件加载问题列表，每行一个，跳过空行"""
    if not os.path.exists(filepath):
        print(f"[错误] 问题文件不存在: {filepath}")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    if not questions:
        print(f"[错误] 问题文件为空: {filepath}")
        sys.exit(1)
    return questions


def chat_completions(
    messages: list,
    api_url: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
):
    """调用 /v1/chat/completions 接口（非流式）"""
    url = api_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    except json.JSONDecodeError as e:
        return {"error": f"JSON 解析失败: {e}"}


def save_results_json(results: list, filepath: str):
    """保存结果为 JSON 文件"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[保存成功] {filepath}")


def save_results_txt(results: list, filepath: str):
    """保存结果为纯文本文件"""
    with open(filepath, "w", encoding="utf-8") as f:
        for idx, item in enumerate(results, 1):
            f.write(f"===== 问题 {idx} =====\n")
            f.write(f"Q: {item['question']}\n\n")
            f.write(f"A: {item['answer']}\n")
            if item.get("error"):
                f.write(f"[错误] {item['error']}\n")
            f.write("\n")
    print(f"[保存成功] {filepath}")


def main():
    if API_URL.endswith("your-api-endpoint.com"):
        print("[错误] 请设置环境变量 API_URL")
        print('  export API_URL="http://your-endpoint.com"')
        sys.exit(1)

    system_prompt = load_system_prompt(SYSTEM_PROMPT_FILE)
    questions = load_questions(QUESTIONS_FILE)

    print(f"[模型] {MODEL}")
    print(f"[接口] {API_URL}/v1/chat/completions")
    print(f"[问题数] {len(questions)}")
    print(f"[输出文件] {OUTPUT_FILE}")
    if system_prompt:
        print(f"[system prompt] 已加载 ({SYSTEM_PROMPT_FILE})")
    print("-" * 50)

    results = []
    start_time = time.time()

    for idx, question in enumerate(questions, 1):
        # 构造单轮消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        print(f"[{idx}/{len(questions)}] 处理中: {question[:60]}{'...' if len(question) > 60 else ''}")

        data = chat_completions(messages, API_URL, MODEL, TEMPERATURE, MAX_TOKENS)

        if "error" in data:
            answer = ""
            error_msg = data["error"]
            print(f"  → [失败] {error_msg}")
        else:
            try:
                answer = data["choices"][0]["message"]["content"]
                error_msg = ""
                print(f"  → [成功] 回复长度: {len(answer)} 字符")
            except (KeyError, IndexError) as e:
                answer = ""
                error_msg = f"响应结构异常: {e}"
                print(f"  → [失败] {error_msg}")

        results.append({
            "index": idx,
            "question": question,
            "answer": answer,
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        })

        if idx < len(questions) and DELAY > 0:
            time.sleep(DELAY)

    elapsed = time.time() - start_time
    print(f"\n[完成] 共处理 {len(questions)} 条，耗时 {elapsed:.1f} 秒")

    # 保存结果
    if OUTPUT_FILE.endswith(".json"):
        save_results_json(results, OUTPUT_FILE)
    else:
        save_results_txt(results, OUTPUT_FILE)

    # 统计
    success_count = sum(1 for r in results if not r["error"])
    fail_count = len(results) - success_count
    print(f"[统计] 成功: {success_count} | 失败: {fail_count}")


if __name__ == "__main__":
    main()
