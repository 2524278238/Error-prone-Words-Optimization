#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对易错词进行两层分类（优先一层：语气词/术语/符号/非源端词/多义词；否则二层：通用词性）
- 使用智谱 GLM-4.5（关闭深度思考：不展示推理、只输出JSON）
- 支持批量、断点续传、结果校验与自动重试
- 目标文件：zhen2_very.json、zhen3_very.json、aren2_very.json、aren3_very.json
- 运行位置：D:/我要毕业/bad_word
输出：同名前缀的结果 JSONL（每行一个词结果），如 zhen2_pos_tags.jsonl
"""

import os
import json
import time
from typing import List, Dict, Any

# 如已安装官方SDK，可使用：from zhipuai import ZhipuAI
# 为兼容无SDK环境，这里提供requests直连实现
import requests

# ============ 配置区域 ============
ZHIPU_API_KEY ='594a382d02834087983824eef3e04f1f.src96KRf9jbwhswg'
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"  # 智谱OpenAPI v4
MODEL_NAME = "glm-4.5"
TEMPERATURE = 0.0
MAX_TOKENS = 50000
BATCH_SIZE = 10 # 每次发送多少个词
RETRY_TIMES = 2
RETRY_SLEEP = 0.5

INPUT_FILES = [
    #"zhen2_very.json",
    # "zhen3_very.json",
    "aren2_very.json",
    "aren3_very.json",
]

# 输出文件命名：{prefix}_pos_tags.jsonl，如 zhen2_pos_tags.jsonl
# 断点续传：若输出已存在，会跳过已完成词

FIRST_SET = ["语气词", "术语", "符号", "非源端词", "多义词"]
SECOND_SET = ["名词", "动词", "形容词", "副词", "数词", "量词", "代词", "介词", "连词", "助词"]

# ============ 工具函数 ============

def load_very_file(path: str) -> List[Dict[str, Any]]:
    """兼容 json 或 jsonl 两种格式的读取（返回列表）"""
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return []
    except json.JSONDecodeError:
        # 尝试按行读取
        items = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return items


def collect_unique_words(records: List[Dict[str, Any]]) -> List[str]:
    words = []
    seen = set()
    for it in records:
        w = it.get("trigger_word") or it.get("word") or ""
        if not w:
            continue
        if w not in seen:
            seen.add(w)
            words.append(w)
    return words


def load_done_words(out_path: str) -> set:
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                w = obj.get("word")
                if w:
                    done.add(w)
            except json.JSONDecodeError:
                continue
    return done


def build_prompt(words: List[str]) -> str:
    # 严格要求JSON数组输出，元素与输入词顺序一一对应
    word_list_str = "\n".join([f"- {w}" for w in words])
    prompt = (
        "你是一个严格的阿拉伯文词性标注器。请对下列词汇进行两层次的词性分类，并严格遵循以下规则：\n"
        "一、第一层类型（优先判断，若符合直接输出最终类别）：\n"
        f"  {FIRST_SET}\n"
        "二、第二层类型（仅当不属于第一层任一类型时，才在此层中选择一个词性）：\n"
        f"  {SECOND_SET}\n"
        "三、判定规范：\n"
        "  1) 先判断词是否属于第一层类型之一；若是，则最终类别=该第一层类型，第二层类别填“无”。\n"
        "  2) 若不属于第一层任何一种，再从第二层类型中选一个最贴切的词性，作为最终类别。\n"
        "  3) 在reason字段中解释或推理过程。\n"
        "四、输入词列表：\n"
        f"{word_list_str}\n\n"
        "五、输出格式（严格的JSON数组，无多余文本）：\n"
        "[ {\"word\": \"<原词>\",\"reason\": \"<词性分类原因>\",\"category\": \"<最终类别>\"} ,\n  ... 与输入顺序一致 ...\n]\n"
        "【必须遵守】只输出JSON，不要任何解释、不要多余字段。"
    )
    return prompt


def call_glm(messages: List[Dict[str, str]]) -> str:
    assert ZHIPU_API_KEY, "请先在环境变量 ZHIPU_API_KEY 中配置智谱API Key"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    resp = requests.post(BASE_URL, headers=headers, data=json.dumps(payload), timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        raise RuntimeError(f"Bad response: {data}")


def parse_and_validate(words: List[str], content: str) -> List[Dict[str, str]]:
    # 尝试截取最外层JSON
    try:
        start = content.find('[')
        end = content.rfind(']')
        if start != -1 and end != -1:
            content = content[start:end+1]
        arr = json.loads(content)
    except Exception:
        raise ValueError("模型输出非有效JSON")

    if not isinstance(arr, list) or len(arr) != len(words):
        raise ValueError("JSON数组长度与输入词数量不一致")

    results = []
    for w, item in zip(words, arr):
        if not isinstance(item, dict):
            raise ValueError("数组元素必须为对象")
        word = item.get("word", "")
        fc1 = item.get("category", "")
        reason = item.get("reason", "")
        # 基础校验
        if word != w:
            # 尝试忽略大小写或空白误差
            if word.strip() != w.strip():
                raise ValueError(f"词顺序/内容不匹配: {word} vs {w}")
        # 约束到合法集合
        if fc1 != "无" and fc1 not in FIRST_SET+SECOND_SET:
            raise ValueError(f"first_category 非法: {fc1}")
        # 规则一致性检查

        results.append({
            "word": w,
            "category": fc1,
            "reason": reason
        })
    return results


def classify_words_for_file(in_path: str):
    prefix = os.path.splitext(os.path.basename(in_path))[0]
    out_path = f"{prefix}_pos_tags.jsonl"

    records = load_very_file(in_path)
    words = collect_unique_words(records)
    if not words:
        print(f"[跳过] 未在 {in_path} 读取到触发词")
        return

    done = load_done_words(out_path)
    todo = [w for w in words if w not in done]
    if not todo:
        print(f"[完成] {in_path} 所有词已处理：{out_path}")
        return

    print(f"开始处理 {in_path}，共{len(words)}个词；待处理{len(todo)}个。输出：{out_path}")

    with open(out_path, 'a', encoding='utf-8') as fout:
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i+BATCH_SIZE]
            prompt = build_prompt(batch)
            messages = [{"role": "user", "content": prompt}]

            attempt = 0
            while True:
                try:
                    content = call_glm(messages)
                    results = parse_and_validate(batch, content)
                    for obj in results:
                        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    print(f"  已处理 {i+len(batch)}/{len(todo)}")
                    break
                except Exception as e:
                    attempt += 1
                    if attempt > RETRY_TIMES:
                        print(f"  [失败] 批次起始索引{i}，错误：{e}")
                        # 将该批次逐词降级处理，尽量不阻塞整体流程
                        for w in batch:
                            single_prompt = build_prompt([w])
                            try:
                                content = call_glm([{"role": "user", "content": single_prompt}])
                                results = parse_and_validate([w], content)
                                fout.write(json.dumps(results[0], ensure_ascii=False) + "\n")
                            except Exception as e2:
                                # 写入一个占位失败结果，便于后续人工补全
                                fallback = {
                                    "word": w,
                                    "category": "无",
                                    "reason": ""
                                }
                                fout.write(json.dumps(fallback, ensure_ascii=False) + "\n")
                                print(f"    [降级] {w}: {e2}")
                        break
                    time.sleep(RETRY_SLEEP)


def main():
    if not ZHIPU_API_KEY:
        print("请先设置环境变量 ZHIPU_API_KEY")
        return

    for path in INPUT_FILES:
        if os.path.exists(path):
            classify_words_for_file(path)
        else:
            print(f"[提示] 未找到文件：{path}，已跳过")

    print("\n全部完成！")

if __name__ == "__main__":
    main()