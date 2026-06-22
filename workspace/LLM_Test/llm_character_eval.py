#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 人设评测批处理脚本
支持单轮独立评测 & 多轮对话评测
"""

import os
import sys
import json
import time
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import requests


class CharacterEvaluator:
    def __init__(
        self,
        api_url: str,
        model_name: str,
        system_prompt_path: str,
        questions_path: str,
        output_dir: str = "./eval_results",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = 60,
        retry_times: int = 3,
        mode: str = "multi",  # "single" 或 "multi"
    ):
        self.api_url = api_url
        self.model_name = model_name
        self.system_prompt_path = Path(system_prompt_path)
        self.questions_path = Path(questions_path)
        self.output_dir = Path(output_dir)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_times = retry_times
        self.mode = mode

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 读取 system prompt
        if not self.system_prompt_path.exists():
            raise FileNotFoundError(f"System prompt 文件不存在: {system_prompt_path}")
        self.system_prompt = self.system_prompt_path.read_text(encoding="utf-8").strip()
        if not self.system_prompt:
            raise ValueError("System prompt 为空")

        # 读取问题列表
        if not self.questions_path.exists():
            raise FileNotFoundError(f"问题列表文件不存在: {questions_path}")
        with open(self.questions_path, "r", encoding="utf-8") as f:
            self.questions = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        if not self.questions:
            raise ValueError("问题列表为空")

        print(f"[初始化] 共加载 {len(self.questions)} 个问题")
        print(f"[初始化] 评测模式: {'多轮对话' if mode == 'multi' else '单轮独立'}")
        print(f"[初始化] 输出目录: {self.output_dir.absolute()}")

    def _call_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """调用 API，带重试机制"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        headers = {
            "Content-Type": "application/json",
        }

        for attempt in range(1, self.retry_times + 1):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                # 解析回复内容
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0].get("message", {}).get("content", "")
                    return {
                        "success": True,
                        "content": content,
                        "raw_response": data,
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    return {
                        "success": False,
                        "content": "",
                        "error": "API 返回格式异常: 无 choices 字段",
                        "raw_response": data,
                        "timestamp": datetime.now().isoformat(),
                    }

            except requests.exceptions.Timeout:
                print(f"    [重试 {attempt}/{self.retry_times}] 请求超时")
                time.sleep(2 * attempt)
            except requests.exceptions.ConnectionError as e:
                print(f"    [重试 {attempt}/{self.retry_times}] 连接错误: {e}")
                time.sleep(2 * attempt)
            except Exception as e:
                return {
                    "success": False,
                    "content": "",
                    "error": str(e),
                    "raw_response": {},
                    "timestamp": datetime.now().isoformat(),
                }

        return {
            "success": False,
            "content": "",
            "error": f"重试 {self.retry_times} 次后仍失败",
            "raw_response": {},
            "timestamp": datetime.now().isoformat(),
        }

    def run_single_mode(self) -> List[Dict[str, Any]]:
        """单轮独立评测：每个问题单独发请求，带 system prompt"""
        results = []
        print(f"\n[单轮模式] 开始评测 {len(self.questions)} 个问题...")

        for idx, question in enumerate(self.questions, 1):
            print(f"  [{idx}/{len(self.questions)}] Q: {question[:50]}...")

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ]

            result = self._call_api(messages)
            result["turn_id"] = idx
            result["question"] = question
            result["mode"] = "single"
            results.append(result)

            if result["success"]:
                print(f"    A: {result['content'][:60]}...")
            else:
                print(f"    [错误] {result.get('error', '未知错误')}")

            # 礼貌延迟，避免限流
            time.sleep(0.5)

        return results

    def run_multi_mode(self) -> List[Dict[str, Any]]:
        """多轮对话评测：维护上下文，问题按顺序进行"""
        results = []
        messages = [{"role": "system", "content": self.system_prompt}]

        print(f"\n[多轮模式] 开始 {len(self.questions)} 轮对话...")

        for idx, question in enumerate(self.questions, 1):
            print(f"  [Turn {idx}/{len(self.questions)}] User: {question[:50]}...")

            messages.append({"role": "user", "content": question})
            result = self._call_api(messages)

            result["turn_id"] = idx
            result["question"] = question
            result["mode"] = "multi"
            results.append(result)

            if result["success"]:
                assistant_content = result["content"]
                messages.append({"role": "assistant", "content": assistant_content})
                print(f"    Assistant: {assistant_content[:60]}...")
            else:
                print(f"    [错误] {result.get('error', '未知错误')}")
                # 即使失败也继续，但不再把错误回复加入上下文

            time.sleep(0.5)

        return results

    def save_results(self, results: List[Dict[str, Any]]) -> None:
        """保存结果到多种格式"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"eval_{self.mode}_{timestamp}"

        # 1. 保存完整 JSON
        json_path = self.output_dir / f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta": {
                        "api_url": self.api_url,
                        "model": self.model_name,
                        "mode": self.mode,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "total_questions": len(self.questions),
                        "system_prompt_length": len(self.system_prompt),
                    },
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n[保存] JSON: {json_path}")

        # 2. 保存 CSV（便于 Excel 查看）
        csv_path = self.output_dir / f"{base_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["turn_id", "mode", "question", "response", "success", "error", "timestamp"])
            for r in results:
                writer.writerow([
                    r["turn_id"],
                    r["mode"],
                    r["question"],
                    r.get("content", ""),
                    r["success"],
                    r.get("error", ""),
                    r["timestamp"],
                ])
        print(f"[保存] CSV: {csv_path}")

        # 3. 保存纯文本对话记录（便于人工阅读）
        txt_path = self.output_dir / f"{base_name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"=== 人设评测记录 ===\n")
            f.write(f"模型: {self.model_name}\n")
            f.write(f"模式: {self.mode}\n")
            f.write(f"时间: {datetime.now().isoformat()}\n")
            f.write(f"问题数: {len(self.questions)}\n")
            f.write("=" * 50 + "\n\n")

            for r in results:
                f.write(f"【Turn {r['turn_id']}】\n")
                f.write(f"User: {r['question']}\n")
                if r["success"]:
                    f.write(f"Assistant: {r['content']}\n")
                else:
                    f.write(f"[ERROR] {r.get('error', '未知错误')}\n")
                f.write("-" * 50 + "\n\n")
        print(f"[保存] TXT: {txt_path}")

    def run(self) -> None:
        """主入口"""
        if self.mode == "single":
            results = self.run_single_mode()
        else:
            results = self.run_multi_mode()

        self.save_results(results)

        # 统计
        success_count = sum(1 for r in results if r["success"])
        print(f"\n[完成] 成功: {success_count}/{len(results)} | 失败: {len(results) - success_count}")
        print(f"[完成] 结果已保存至: {self.output_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="LLM 人设评测批处理脚本")
    parser.add_argument("--api-url", required=True, help="API 地址，如 http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", required=True, help="模型名称")
    parser.add_argument("--system-prompt", required=True, help="System prompt 文件路径")
    parser.add_argument("--questions", required=True, help="问题列表文件路径（每行一个）")
    parser.add_argument("--output", default="./eval_results", help="输出目录")
    parser.add_argument("--mode", choices=["single", "multi"], default="multi", help="评测模式: single=单轮独立, multi=多轮对话（默认）")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retry", type=int, default=3)

    args = parser.parse_args()

    evaluator = CharacterEvaluator(
        api_url=args.api_url,
        model_name=args.model,
        system_prompt_path=args.system_prompt,
        questions_path=args.questions,
        output_dir=args.output,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retry_times=args.retry,
        mode=args.mode,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
