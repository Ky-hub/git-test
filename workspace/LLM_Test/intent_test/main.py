import json
import os
import re
from typing import Dict, Any


INTENT_CONFIG_FILE = os.getenv("INTENT_CONFIG_FILE", "intent_config.json")


def load_intent_config(path: str = INTENT_CONFIG_FILE) -> dict:
    """加载意图配置文件"""
    if not os.path.exists(path):
        return {"intents": [], "fallback_intent": "unknown"}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recognize_intent(text: str, config_path: str = INTENT_CONFIG_FILE) -> Dict[str, Any]:
    """
    纯 LLM 意图识别（无规则匹配）
    
    返回: {"intent": str, "confidence": float}
    """
    config = load_intent_config(config_path)
    intents = config.get("intents", [])
    fallback = config.get("fallback_intent", "unknown")

    if not intents:
        return {"intent": fallback, "confidence": 0.0}

    # 构建意图描述文本
    lines = []
    for intent in intents:
        name = intent["name"]
        desc = intent.get("description", "")
        examples = intent.get("examples", [])
        ex_str = f" 示例：{', '.join(examples[:3])}" if examples else ""
        lines.append(f"- {name}: {desc}{ex_str}")

    system_prompt = (
        "你是一个意图识别专家。请分析用户输入，判断属于以下哪个意图：\n\n"
        + "\n".join(lines)
        + f'\n\n如果无法匹配任何意图，请返回意图名称："{fallback}"\n'
        "请严格按以下 JSON 格式返回，不要输出任何其他内容：\n"
        '{"intent": "意图名称", "confidence": 0.95}'
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    # 调用 LLM（复用你现有的 chat_completions，非流式）
    response = chat_completions(
        messages=messages,
        api_url=API_URL,
        model=MODEL,
        stream=False,
        temperature=0.1,
        max_tokens=128,
    )

    if response is None:
        return {"intent": fallback, "confidence": 0.0}

    try:
        content = response.json()["choices"][0]["message"]["content"]
        # 提取 JSON 块
        match = re.search(r"\{.*?\}", content, re.DOTALL)
        if match:
            result = json.loads(match.group())
            return {
                "intent": result.get("intent", fallback),
                "confidence": float(result.get("confidence", 0.0)),
            }
    except Exception:
        pass

    return {"intent": fallback, "confidence": 0.0}
