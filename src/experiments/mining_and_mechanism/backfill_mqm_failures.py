#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 mqm_error_classifier 生成的结果文件进行补充：
针对 'API调用失败' 的条目，使用备用 API (shubiaobiao OpenAI 兼容接口) 重新请求，
并输出补充后的新结果文件（默认在同目录生成 *_filled.jsonl）。
"""

import os
import json
import time
import argparse
import requests
import ast


def extract_json_from_response(response_text: str):
    """从模型响应中提取 JSON（与原脚本逻辑保持一致的鲁棒处理）。"""
    try:
        response_text = response_text.replace('```json\n', '').replace('\n```', '')
        if response_text.strip().startswith('{'):
            return json.loads(response_text.strip())

        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)

        return None
    except json.JSONDecodeError:
        try:
            txt = response_text
            txt = txt.replace('```json', '').replace('```', '').strip()
            start = txt.find('{')
            end = txt.rfind('}')
            if start != -1 and end != -1 and end > start:
                txt = txt[start:end + 1]

            key = '"error_description"'
            kpos = txt.find(key)
            if kpos != -1:
                vstart = txt.find('"', kpos + len(key))
                if vstart != -1:
                    cpos = txt.find('"confidence"')
                    search_upto = cpos if cpos != -1 else len(txt)
                    end_quote = -1
                    scan = vstart + 1
                    while scan < search_upto:
                        qpos = txt.find('"', scan)
                        if qpos == -1 or qpos >= search_upto:
                            break
                        if qpos + 1 < len(txt) and txt[qpos + 1] == ',':
                            end_quote = qpos
                            break
                        if txt[qpos - 1] == '\\':
                            scan = qpos + 1
                            continue
                        scan = qpos + 1
                    if end_quote != -1:
                        val = txt[vstart + 1:end_quote]
                        val_fixed = val.replace('"', "'")
                        txt = txt[:vstart + 1] + val_fixed + txt[end_quote:]
            return json.loads(txt)
        except Exception:
            return None

MQM_CATEGORIES_ZH = {
    "准确性 (Accuracy)": {
        "误译 (Mistranslation)": "译文传达了与原文不同的含义。",
        "漏译 (Omission)": "译文遗漏了原文的信息。",
        "增译 (Addition)": "译文包含了原文没有的信息。",
        "未翻译 (Untranslated)": "原文的部分内容在没有正当理由的情况下被保留未翻译。"
    },
    "流畅性 (Fluency)": {
        "语法 (Grammar)": "目标语言的语法结构不正确。",
        "词序 (Word Order)": "词语或短语的顺序不符合目标语言的规范。",
        "功能词 (Function Words)": "冠词、介词或其他功能词使用不当或缺失。",
        "拼写 (Spelling)": "单词拼写错误。",
        "标点 (Punctuation)": "标点符号使用不当、缺失或不合适。",
        "字符编码 (Character Encoding)": "由于编码问题导致文本乱码或字符不正确。"
    },
    "风格 (Style)": {
        "语域 (Register)": "正式程度或语调不适合上下文。",
        "风格不一致 (Inconsistent Style)": "文本整体风格不一致。"
    },
    "术语 (Terminology)": {
        "术语误译 (Term Mistranslation)": "领域特定的术语翻译不正确。",
        "术语不一致 (Inconsistent Term)": "同一术语在文本中翻译不一致。"
    },
    "本地化规范 (Locale Conventions)": {
        "日期/时间/数字/货币格式": "日期、时间、数字或货币的格式不符合目标语言或地区的规范。",
        "单位 (Units)": "计量单位不正确、未本地化或未适当转换。"
    },
    "其他 (Other)": {
        "其他 (Other)": "不属于以上任何类别的质量问题。"
    },
    "无错误 (No Error)": {
        "无错误 (No Error)": "翻译准确流畅。"
    }
}


def build_mqm_prompt(src_text, mt_text, ref_text, src_language, comet_score, avg_comet):
    """构建MQM错误分类提示（与 mqm_error_classifier.py 保持一致）"""

    quality_hint = "质量提示：COMET分数信息不足。"
    try:
        delta = float(comet_score) - float(avg_comet)
        if delta >= 0.05:
            quality_hint = f"质量提示：该句COMET显著高于平均（Δ={delta:+.4f}），整体质量较高；如未发现明确错误，请选择‘无错误-无错误’。"
        elif delta <= -0.05:
            quality_hint = f"质量提示：该句COMET显著低于平均（Δ={delta:+.4f}），整体质量可能较差；请谨慎检查主要错误。"
        else:
            quality_hint = f"质量提示：该句COMET接近平均（Δ={delta:+.4f}），请基于内容判断是否存在明显错误。"
    except Exception:
        pass

    categories_str = ""
    for major, minor_map in MQM_CATEGORIES_ZH.items():
        categories_str += f"\n{major}:\n"
        for minor, desc in minor_map.items():
            categories_str += f"- {minor}: {desc}\n"

    prompt = f"""你是一个专业的机器翻译质量评估专家，请根据MQM(多维质量度量)标准对以下翻译进行错误分类。

### 上下文信息
- 源语言: {src_language}
- 目标语言: 英文
- 句子 COMET 分数: {comet_score:.4f}
- 全体平均 COMET 分数: {avg_comet:.4f}
- {quality_hint}

### 文本内容
- 原文: {src_text}
- 机器翻译: {mt_text}
- 参考翻译: {ref_text}

### 任务
请分析“机器翻译”中的主要错误，并从下面的错误类别中选择最主要的一个。

### 错误类别
{categories_str}

### 输出格式
请务必仅使用以下JSON格式提供您的分析，不要在JSON结构之外添加任何解释；不要使用代码块；不要在任何value中使用双引号（如需引用用单引号）。
{{
    "error_category": "大类-子类 (例如: 准确性-误译)",
    "severity": "轻微/主要/严重",
    "error_description": "对主要错误的简明描述。",
    "confidence": "高/中/低"
}}

### 指示
1. 如果存在多个错误，请选择对翻译质量影响最大的一个。
2. 若整体质量较高且无明显错误，请选择“无错误-无错误”。
"""
    return prompt


def infer_lang_pair_from_path(in_path: str) -> str:
    d = os.path.dirname(os.path.abspath(in_path))
    return os.path.basename(d)


def get_lang_config(lang_pair: str):
    if lang_pair == 'zhen2':
        return {
            'comet_file': os.path.join('bad_word', 'zhen2', '119w.comet'),
            'src_lang': '中文'
        }
    if lang_pair == 'zhen3':
        return {
            'comet_file': os.path.join('bad_word', 'zhen3', '119w.comet'),
            'src_lang': '中文'
        }
    if lang_pair == 'aren2':
        return {
            'comet_file': os.path.join('bad_word', 'aren2', '99w.comet'),
            'src_lang': '阿拉伯文'
        }
    if lang_pair == 'aren3':
        return {
            'comet_file': os.path.join('bad_word', 'aren3', '99w.comet'),
            'src_lang': '阿拉伯文'
        }
    return None


def load_comet_scores(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = ast.literal_eval(f.read())
    # 与原脚本一致：仅取第二列为分值
    return [row[1] for row in data]


class FallbackClient:
    """备用 API 客户端（shubiaobiao OpenAI 兼容接口）。"""

    def __init__(self, api_key: str, base_url: str = "https://api.shubiaobiao.cn/v1", model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()

    def chat(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0, timeout: int = 60) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = self.session.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"fallback API failed: {resp.status_code} {resp.text}")
        data = resp.json()
        if not data.get('choices'):
            raise RuntimeError(f"fallback API no choices: {data}")
        return data['choices'][0]['message']['content']


def backfill_failed_entries(input_file: str, output_file: str, api_key: str, sleep_sec: float = 0.2, limit: int = None):
    """对 input_file 中 'API调用失败' 的条目使用备用 API 进行补充，写入 output_file。"""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"未找到结果文件: {input_file}")

    client = FallbackClient(api_key=api_key)

    # 推断语言对并加载 COMET 分数与均值
    lang_pair = infer_lang_pair_from_path(input_file)
    cfg = get_lang_config(lang_pair)
    comet_scores = None
    avg_comet = None
    src_language = '中文' if lang_pair.startswith('zh') else '阿拉伯文'
    if cfg and os.path.exists(cfg['comet_file']):
        try:
            comet_scores = load_comet_scores(cfg['comet_file'])
            avg_comet = sum(comet_scores) / len(comet_scores) if comet_scores else None
            src_language = cfg['src_lang']
        except Exception:
            comet_scores = None
            avg_comet = None

    total = 0
    fixed = 0
    kept = 0
    attempted = 0

    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                item = json.loads(line)
            except Exception:
                # 跳过无法解析的行
                continue

            mqm = item.get('mqm_analysis') or {}
            is_failed = isinstance(mqm, dict) and (mqm.get('错误') == 'API调用失败' or mqm.get('错误') == 'JSON解析失败')

            if is_failed:
                # 若达到尝试上限，则不再调用备用接口，直接原样写入
                if limit is not None and attempted >= limit:
                    kept += 1
                    fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                    continue

                attempted += 1
                src_text = item.get('src_text', '')
                mt_text = item.get('mt_text', '')
                ref_text = item.get('ref_text', '')

                # 取与原脚本一致的提示词（包含 COMET 分数与均值）
                idx = item.get('sentence_index', 0) or 0
                c_score = None
                if isinstance(idx, int) and comet_scores and 0 <= idx < len(comet_scores):
                    c_score = comet_scores[idx]
                # 若无法获得有效分数，使用均值或 0 填充，保持模板不变
                if c_score is None:
                    c_score = avg_comet if isinstance(avg_comet, (int, float)) else 0.0
                a_score = avg_comet if isinstance(avg_comet, (int, float)) else 0.0
                prompt = build_mqm_prompt(src_text, mt_text, ref_text, src_language, c_score, a_score)
                raw = None
                parsed = None
                error = None
                try:
                    raw = client.chat(prompt, max_tokens=256, temperature=0.0, timeout=60)
                    parsed = extract_json_from_response(raw)
                    if parsed is None:
                        error = '备用API JSON解析失败'
                except Exception as e:
                    error = f'备用API调用失败: {e}'

                if parsed is not None:
                    item['mqm_analysis'] = parsed
                    item['used_fallback'] = True
                    item['raw_response_fallback'] = raw
                    fixed += 1
                else:
                    item['fallback_error'] = error
                    item['raw_response_fallback'] = raw
                    kept += 1

                time.sleep(sleep_sec)
            else:
                kept += 1

            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    return {"total": total, "attempted": attempted, "fixed": fixed, "kept": kept, "output": output_file}


def infer_default_paths(input_path: str = None):
    """推断默认输入/输出路径。"""
    if input_path:
        in_path = input_path
    else:
        in_path = os.path.join('bad_word', 'zhen2', 'mqm_classification_results.jsonl')
    out_path = os.path.splitext(in_path)[0] + '_filled.jsonl'
    return in_path, out_path


def main():
    parser = argparse.ArgumentParser(description='补充 MQM 结果文件中 API 失败的条目')
    parser.add_argument('--input', type=str, default=None, help='输入结果文件（jsonl），默认 bad_word/zhen3/mqm_classification_results.jsonl')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径，默认在输入文件名后追加 _filled.jsonl')
    parser.add_argument('--api_key', type=str, default='sk-Hubl0HXRiL8Tln1a0c22FaD02fA84862B211Cc1b92F7B97e', help='备用接口 API Key')
    parser.add_argument('--sleep', type=float, default=1, help='每次调用后的休眠秒数')
    parser.add_argument('--limit', type=int, default=None, help='最多处理的行数（调试用）')
    args = parser.parse_args()

    in_path, default_out = infer_default_paths(args.input)
    out_path = args.output or default_out

    stats = backfill_failed_entries(in_path, out_path, api_key=args.api_key, sleep_sec=args.sleep, limit=args.limit)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()


