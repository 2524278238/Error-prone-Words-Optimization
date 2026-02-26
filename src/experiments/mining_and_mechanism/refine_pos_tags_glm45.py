#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新判定pos标签：仅对已有结果中 `category` == "无" 且 `reason` 为空 的词进行补充分类。
- 中文集（zhen2/zhen3）使用：“你是一个严格的中文词性标注器。”
- 阿语集（aren2/aren3）使用：“你是一个严格的阿拉伯文词性标注器。”
- 输出到 *_fixed.jsonl，不覆盖原文件
"""

import os
import json
import time
import requests
from typing import List, Dict, Any

# 配置
ZHIPU_API_KEY ='594a382d02834087983824eef3e04f1f.src96KRf9jbwhswg'
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL_NAME = "glm-4.5"
TEMPERATURE = 0.0
MAX_TOKENS = 4096
BATCH_SIZE = 20
RETRY_TIMES = 2
RETRY_SLEEP = 0.5

FIRST_SET = ["语气词", "术语", "符号", "非源端词", "多义词"]
SECOND_SET = ["名词", "动词", "形容词", "副词", "数词", "量词", "代词", "介词", "连词", "助词"]

INPUT_FILES = [
    os.path.join("bad_word", "zhen2", "zhen2_very_pos_tags.jsonl"),
    os.path.join("bad_word", "zhen3", "zhen3_very_pos_tags.jsonl"),
    os.path.join("bad_word", "aren2", "aren2_very_pos_tags.jsonl"),
    os.path.join("bad_word", "aren3", "aren3_very_pos_tags.jsonl"),
]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def save_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, 'w', encoding='utf-8') as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def is_arabic_file(path: str) -> bool:
    name = os.path.basename(path)
    return name.startswith("aren")


def build_prompt(words: List[str], arabic: bool) -> str:
    head = "你是一个严格的阿拉伯文词性标注器。" if arabic else "你是一个严格的中文词性标注器。"
    word_list_str = "\n".join([f"- {w}" for w in words])
    prompt = (
        f"{head}请对下列词汇进行两层次的词性分类，并严格遵循以下规则：\n"
        "一、第一层类型（优先判断，若符合直接输出最终类别）：\n"
        f"  {FIRST_SET}\n"
        "二、第二层类型（仅当不属于第一层任一类型时，才在此层中选择一个词性）：\n"
        f"  {SECOND_SET}\n"
        "三、判定规范：\n"
        "  1) 先判断词是否属于第一层类型之一；若是，则最终类别=该第一层类型。\n"
        "  2) 若不属于第一层任何一种，再从第二层类型中选一个最贴切的词性，作为最终类别。\n"
        "  3) 在reason字段中简要说明判定依据。\n"
        "四、输入词列表：\n"
        f"{word_list_str}\n\n"
        "五、输出格式（严格的JSON数组，无多余文本）：\n"
        "[ {\"word\": \"<原词>\",\"reason\": \"<原因>\",\"category\": \"<最终类别>\"},\n  ... 与输入顺序一致 ... ]\n"
        "【必须遵守】只输出JSON，不要任何解释、不要多余字段。"
    )
    return prompt


def call_glm(messages: List[Dict[str, str]]) -> str:
    assert ZHIPU_API_KEY, "请先设置环境变量 ZHIPU_API_KEY"
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


def parse_response(words: List[str], content: str) -> List[Dict[str, str]]:
    # 截取最外层JSON数组
    start = content.find('[')
    end = content.rfind(']')
    if start != -1 and end != -1:
        content = content[start:end+1]
    arr = json.loads(content)
    if not isinstance(arr, list) or len(arr) != len(words):
        raise ValueError("JSON数组长度与输入词数量不一致")
    out = []
    for w, item in zip(words, arr):
        if not isinstance(item, dict):
            raise ValueError("数组元素必须为对象")
        ww = item.get("word", "")
        cat = item.get("category", "")
        reason = item.get("reason", "")
        if ww.strip() != w.strip():
            raise ValueError(f"词顺序/内容不匹配: {ww} vs {w}")
        if not cat or (cat not in FIRST_SET + SECOND_SET and cat != "无"):
            raise ValueError(f"非法类别: {cat}")
        out.append({"word": w, "category": cat, "reason": reason})
    return out


def refine_file(path: str):
    records = load_jsonl(path)
    if not records:
        print(f"[提示] 空文件或不存在：{path}")
        return

    # 找出需要补充的词
    todo_words = []
    for obj in records:
        if obj.get("category") == "无" and (not obj.get("reason")):
            w = obj.get("word")
            if w and w not in todo_words:
                todo_words.append(w)

    if not todo_words:
        print(f"[完成] {path} 无需补充分类")
        return

    arabic = is_arabic_file(path)
    print(f"开始补充分类：{path}（{len(todo_words)} 个词），arabic={arabic}")

    # 批量请求
    updated = {}
    for i in range(0, len(todo_words), BATCH_SIZE):
        batch = todo_words[i:i+BATCH_SIZE]
        prompt = build_prompt(batch, arabic)
        messages = [{"role": "user", "content": prompt}]

        attempt = 0
        while True:
            try:
                content = call_glm(messages)
                parsed = parse_response(batch, content)
                for item in parsed:
                    updated[item["word"]] = item
                print(f"  已补充 {i+len(batch)}/{len(todo_words)}")
                break
            except Exception as e:
                attempt += 1
                if attempt > RETRY_TIMES:
                    print(f"  [失败] 批次{i}, 错误：{e}")
                    # 降级逐词
                    for w in batch:
                        sprompt = build_prompt([w], arabic)
                        try:
                            scontent = call_glm([{"role": "user", "content": sprompt}])
                            sparsed = parse_response([w], scontent)
                            updated[w] = sparsed[0]
                        except Exception as e2:
                            updated[w] = {"word": w, "category": "无", "reason": ""}
                            print(f"    [降级] {w}: {e2}")
                    break
                time.sleep(RETRY_SLEEP)

    # 生成fixed文件
    out_path = path.replace(".jsonl", "_fixed.jsonl")
    new_records = []
    for obj in records:
        w = obj.get("word")
        if w in updated:
            obj["category"] = updated[w].get("category", obj.get("category", "无"))
            obj["reason"] = updated[w].get("reason", obj.get("reason", ""))
        new_records.append(obj)
    save_jsonl(out_path, new_records)
    print(f"[输出] 已写入：{out_path}")


def main():
    if not ZHIPU_API_KEY:
        print("请先设置环境变量 ZHIPU_API_KEY")
        return

    for path in INPUT_FILES:
        refine_file(path)

    print("\n全部完成！")

if __name__ == "__main__":
    main()
