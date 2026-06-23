#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图分类 AI 助手（按 Category 识别）
纯 LLM 语义判断，对话形式输出分类结果
"""

import os
import sys
import json
import requests


# ==================== 配置区域 ====================
API_URL = os.getenv("API_URL", "https://your-api-endpoint.com")
MODEL = os.getenv("MODEL", "qwen3.6-27b")

# 分类配置文件路径
CATEGORY_CONFIG_FILE = os.getenv("CATEGORY_CONFIG_FILE", "category_config.json")
# 额外指令文件（可选）
EXTRA_PROMPT_FILE = os.getenv("EXTRA_PROMPT_FILE", "extra_prompt.txt")

STREAM = os.getenv("STREAM", "true").lower() in ("true", "1", "yes")
TIMEOUT = int(os.getenv("TIMEOUT", "60"))
# =================================================


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[警告] 未找到配置文件: {path}")
        return {"categories": [], "fallback_category": "普通对话"}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_extra_prompt(path: str) -> str:
    if os.path.exists(path):
        return open(path, "r", encoding="utf-8").read().strip()
    return ""


def build_system_prompt(config: dict, extra: str = "") -> str:
    """
    构造结构化 Prompt，识别目标是 Category 而非具体 Intent
    """
    categories = config.get("categories", [])
    fallback = config.get("fallback_category", "普通对话")

    # 构建分类定义区块
    blocks = []
    for idx, cat in enumerate(categories, 1):
        name = cat.get("name", "")
        desc = cat.get("description", "")
        examples = cat.get("examples", [])
        intents = cat.get("intents", [])

        # 分类下的具体意图（辅助模型理解该分类的覆盖范围）
        intent_lines = []
        if intents:
            for intent in intents:
                i_name = intent.get("name", "")
                i_desc = intent.get("description", "")
                i_ex = intent.get("examples", [])
                i_ex_str = f" 例如：{', '.join(i_ex[:2])}" if i_ex else ""
                intent_lines.append(f"      · {i_name}：{i_desc}{i_ex_str}")
            intent_block = "\n" + "\n".join(intent_lines)
        else:
            intent_block = ""

        # 典型表达示例
        ex_str = ""
        if examples:
            quoted = [f'"{ex}"' for ex in examples[:3]]
            ex_str = f"\n    典型表达：{', '.join(quoted)}"

        block = (
            f"{idx}. 【分类】{name}\n"
            f"    定义：{desc}"
            f"{ex_str}"
            f"{intent_block}"
        )
        blocks.append(block)

    catalog = "\n\n".join(blocks) if blocks else "（暂无预定义分类）"

    # Few-shot 教模型以分类维度回答
    few_shots = """【判断示例】

用户输入："今天北京天气怎么样？"
你的回应：我判断您的意图分类是【信息查询】。您想获取天气相关信息，需要我帮您查询具体数据吗？

用户输入："你好"
你的回应：我判断您的意图分类是【基础交互】。很高兴见到您，有什么我可以帮您的吗？

用户输入："帮我订一张去上海的机票"
你的回应：我判断您的意图分类是【业务办理】。您有预订机票的需求，请告诉我出发时间和具体日期。

用户输入："随便聊聊"
你的回应：我判断您的意图分类是【基础交互】。没问题，我们可以随便聊聊，您最近有什么新鲜事吗？"""

    prompt = f"""你是一个专业的意图分类助手。请根据用户的输入，从以下分类列表中选择最匹配的一个，并以自然、友好的对话形式告知用户。

=== 分类定义表 ===
{catalog}

【兜底规则】
如果用户输入无法归入以上任何分类，请将其识别为「{fallback}」。

{few_shots}

【输出要求】
1. 直接以对话形式回复用户，说明识别到的分类名称和简要理由；
2. 语气友好、简洁，像助手一样说话；
3. 不要输出 JSON、代码块或结构化数据，只输出自然语言；
4. 如果用户输入属于普通对话/闲聊，请正常回应并说明分类为「{fallback}」；
5. 如果用户输入包含多个分类特征，请识别最主要的一个并说明。
"""

    if extra:
        prompt += f"\n=== 额外指令 ===\n{extra}\n"

    return prompt


def chat_completions(messages, api_url, model, stream=True, temperature=0.3, max_tokens=512):
    url = api_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
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


def parse_stream(response):
    full = ""
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            s = line[len("data: "):]
            if s.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(s)
                content = chunk["choices"][0].get("delta", {}).get("content", "")
                if content:
                    print(content, end="", flush=True)
                    full += content
            except:
                continue
    print()
    return full


def parse_non_stream(response):
    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        print(content)
        return content
    except Exception as e:
        print(f"[解析错误] {e}")
        return ""


def main():
    if API_URL.endswith("your-api-endpoint.com"):
        print("[错误] 请设置环境变量 API_URL")
        print('  export API_URL="https://api.example.com"')
        sys.exit(1)

    config = load_config(CATEGORY_CONFIG_FILE)
    extra = load_extra_prompt(EXTRA_PROMPT_FILE)
    system_prompt = build_system_prompt(config, extra)

    print(f"[模型] {MODEL}")
    print(f"[接口] {API_URL}/v1/chat/completions")
    print(f"[模式] {'流式' if STREAM else '非流式'}")
    print(f"[分类配置] {CATEGORY_CONFIG_FILE} ({len(config.get('categories', []))} 个分类)")
    print("-" * 50)
    print("我是意图分类助手，请告诉我你想做什么，我会帮你判断所属分类。\n")
    print("输入 'exit' 或 'quit' 退出，输入 'reload' 重新加载配置\n")

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
        if user_input.lower() == "reload":
            config = load_config(CATEGORY_CONFIG_FILE)
            extra = load_extra_prompt(EXTRA_PROMPT_FILE)
            system_prompt = build_system_prompt(config, extra)
            print("[配置已重新加载]")
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        print("Assistant: ", end="", flush=True)
        resp = chat_completions(messages, API_URL, MODEL, stream=STREAM)

        if resp is None:
            continue

        if STREAM:
            parse_stream(resp)
        else:
            parse_non_stream(resp)

        print()


if __name__ == "__main__":
    main()
