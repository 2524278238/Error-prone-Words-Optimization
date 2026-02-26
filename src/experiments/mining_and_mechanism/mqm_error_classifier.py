#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于MQM错误分类的翻译错误原因分析
使用GLM-4.5对每个翻译句对进行错误分类判断
"""

import json
import requests
import time
import os,ast
from collections import defaultdict, Counter

# API配置
ZHIPU_API_KEY = '594a382d02834087983824eef3e04f1f.src96KRf9jbwhswg'
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL_NAME = "glm-4.5"

# MQM错误分类 (中文版)
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


def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return []
    return data

def load_json(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return []

def load_text_lines(file_path):
    """加载文本文件按行"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return []

def load_comet_scores(file_path):
    """加载COMET分数文件"""
    with open(file_path,'r')as f:
        a=ast.literal_eval(f.read())
    return a

def call_glm45_api(prompt, max_tokens=4096):
    """调用GLM-4.5 API"""
    headers = {
        'Authorization': f'Bearer {ZHIPU_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(BASE_URL, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'].strip()
        
        print(f"API调用失败: {response.status_code}, {response.text}")
        return None
        
    except Exception as e:
        print(f"API调用异常: {e}")
        return None

def build_mqm_prompt(src_text, mt_text, ref_text, src_language, comet_score, avg_comet):
    """构建MQM错误分类提示（不包含触发词，加入COMET质量引导）"""

    # 基于COMET差值的质量引导
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

def extract_json_from_response(response_text):
    """从响应中提取JSON"""
    try:
        # 尝试直接解析
        response_text=response_text.replace('```json\n','').replace('\n```','')
        if response_text.strip().startswith('{'):
            return json.loads(response_text.strip())
        
        # 查找JSON块
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
        
        return None
        
    except json.JSONDecodeError:
        # 二次修复：移除代码块包裹，并尽量修正 error_description 内部的未转义双引号
        try:
            txt = response_text
            txt = txt.replace('```json', '').replace('```', '').strip()
            # 仅截取最外层花括号内容
            start = txt.find('{')
            end = txt.rfind('}')
            if start != -1 and end != -1 and end > start:
                txt = txt[start:end+1]
            # 尝试定位 error_description，并替换其中未转义的双引号为单引号
            key = '"error_description"'
            kpos = txt.find(key)
            if kpos != -1:
                # 从 key 后的第一个引号开始
                vstart = txt.find('"', kpos + len(key))
                if vstart != -1:
                    # 以 "confidence" 作为锚点，向前寻找 error_description 结束引号
                    cpos = txt.find('"confidence"')
                    search_upto = cpos if cpos != -1 else len(txt)
                    # 在 (vstart, search_upto) 区间内寻找以 ", 结束的引号
                    end_quote = -1
                    scan = vstart + 1
                    while scan < search_upto:
                        qpos = txt.find('"', scan)
                        if qpos == -1 or qpos >= search_upto:
                            break
                        # 判断是否为结束引号：下一个非空白字符是否为逗号或换行，并且后面不在转义状态
                        # 简化：如果后面紧跟逗号，则认为是结束引号
                        if qpos + 1 < len(txt) and txt[qpos + 1] == ',':
                            end_quote = qpos
                            break
                        # 跳过被反斜杠转义的引号
                        if txt[qpos - 1] == '\\':
                            scan = qpos + 1
                            continue
                        # 否则继续寻找
                        scan = qpos + 1
                    if end_quote != -1:
                        val = txt[vstart+1:end_quote]
                        # 将未转义的双引号替换为单引号
                        val_fixed = val.replace('"', "'")
                        txt = txt[:vstart+1] + val_fixed + txt[end_quote:]
            return json.loads(txt)
        except Exception:
            print(f"JSON解析失败: {response_text}")
            return None

def build_sentence_data(data_dir, language_pair):
    """构建句子数据"""
    print(f"构建 {language_pair} 的句子数据...")
    
    # 确定文件路径
    if language_pair == 'zhen2':
        alignment_file = 'bad_word/zhen2/word_alignment_results_jieba_tgt.jsonl'
        ref_alignment_file = 'bad_word/zhen2/word_alignment_results_jieba_ref.jsonl'
        comet_file = 'bad_word/zhen2/119w.comet'
        very_file = 'bad_word/zhen2_very.json'
        src_lang = "中文"
    elif language_pair == 'zhen3':
        alignment_file = 'bad_word/zhen3/word_alignment_results_jieba_tgt.jsonl'
        ref_alignment_file = 'bad_word/zhen3/word_alignment_results_jieba_ref.jsonl'
        comet_file = 'bad_word/zhen3/119w.comet'
        very_file = 'bad_word/zhen3_very.json'
        src_lang = "中文"
    elif language_pair == 'aren2':
        alignment_file = 'bad_word/aren2/word_alignment_results_jieba_tgt.jsonl'
        ref_alignment_file = 'bad_word/aren2/word_alignment_results_jieba_ref.jsonl'
        comet_file = 'bad_word/aren2/99w.comet'
        very_file = 'bad_word/aren2_very.json'
        src_lang = "阿拉伯文"
    elif language_pair == 'aren3':
        alignment_file = 'bad_word/aren3/word_alignment_results_jieba_tgt.jsonl'
        ref_alignment_file = 'bad_word/aren3/word_alignment_results_jieba_ref.jsonl'
        comet_file = 'bad_word/aren3/99w.comet'
        very_file = 'bad_word/aren3_very.json'
        src_lang = "阿拉伯文"
    else:
        raise ValueError(f"不支持的语言对: {language_pair}")
    
    # 加载数据
    print("加载文件...")
    tgt_alignments = load_jsonl(alignment_file)
    ref_alignments = load_jsonl(ref_alignment_file)
    comet_scores = load_comet_scores(comet_file)
    comet_scores=[i[1] for i in comet_scores]
    very_words_list = load_jsonl(very_file) # 修复: very.json是jsonl格式
    
    if not all([tgt_alignments, ref_alignments, very_words_list]):
        print("一个或多个关键数据文件加载失败! (tgt_alignments, ref_alignments, very_words_list)")
        return [], src_lang
    avg_comet = sum(comet_scores)/len(comet_scores) 
    print(f"文件长度: tgt_alignments={len(tgt_alignments)}, ref_alignments={len(ref_alignments)}, comet_scores={len(comet_scores)}")

    # 从词对齐数据构建句子数据
    sentence_data = []
    
    # 使用字典来去重，确保每个(sentence_index, trigger_word)组合只出现一次
    seen_combinations = set()
    
    for tgt_item in tgt_alignments:
        sentence_index = tgt_item.get('index')
        trigger_word = tgt_item.get('trigger_word')
        src_sentence = tgt_item.get('src_sentence', '')
        tgt_sentence = tgt_item.get('tgt_sentence', '')
        
        # 查找对应的参考对齐
        ref_sentence = ''
        for ref_item in ref_alignments:
            if (ref_item.get('index') == sentence_index and 
                ref_item.get('trigger_word') == trigger_word):
                ref_sentence = ref_item.get('tgt_sentence', '')  # ref文件中的tgt_sentence是参考翻译
                break
        
        # 去重检查
        combination_key = (sentence_index, trigger_word)
        if combination_key in seen_combinations:
            continue
        seen_combinations.add(combination_key)
        
        if src_sentence and tgt_sentence:
            sentence_data.append({
                'sentence_index': sentence_index,
                'trigger_word': trigger_word,
                'src_text': src_sentence,
                'mt_text': tgt_sentence,
                'ref_text': ref_sentence,
                'src_language': src_lang,
                'comet_score': comet_scores[sentence_index],
                'avg_comet': avg_comet
            })
    
    print(f"构建完成，共 {len(sentence_data)} 个句子")
    return sentence_data, src_lang

def process_sentences(sentence_data, output_file, resume_from=0):
    """处理句子进行MQM分类"""
    print(f"开始处理句子，从第 {resume_from} 个开始...")
    
    results = []
    
    # 如果输出文件存在，先加载已有结果
    if os.path.exists(output_file) and resume_from > 0:
        print("加载已有结果...")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line.strip()))
            print(f"已加载 {len(results)} 个结果")
        except Exception as e:
            print(f"加载已有结果失败: {e}")
            results = []
    
    # 处理剩余句子
    for i in range(resume_from, len(sentence_data)):
        sentence = sentence_data[i]
        
        print(f"处理进度: {i+1}/{len(sentence_data)} - 触发词: {sentence['trigger_word']}")
        
        # 构建提示（移除触发词）
        prompt = build_mqm_prompt(
            sentence['src_text'],
            sentence['mt_text'], 
            sentence['ref_text'],
            sentence['src_language'],
            sentence['comet_score'],
            sentence['avg_comet']
        )
        if i<5:
            print(prompt)
        # 调用API
        response = call_glm45_api(prompt)
        
        if response:
            # 解析JSON响应
            parsed_result = extract_json_from_response(response)
            
            if parsed_result:
                result = {
                    'sentence_index': sentence['sentence_index'],
                    'trigger_word': sentence['trigger_word'],
                    'src_text': sentence['src_text'],
                    'mt_text': sentence['mt_text'],
                    'ref_text': sentence['ref_text'],
                    'mqm_analysis': parsed_result,
                    'raw_response': response
                }
            else:
                result = {
                    'sentence_index': sentence['sentence_index'],
                    'trigger_word': sentence['trigger_word'],
                    'src_text': sentence['src_text'],
                    'mt_text': sentence['mt_text'],
                    'ref_text': sentence['ref_text'],
                    'mqm_analysis': {"错误": "JSON解析失败"},
                    'raw_response': response
                }
            if i<5:
                print(response)
        else:
            result = {
                'sentence_index': sentence['sentence_index'],
                'trigger_word': sentence['trigger_word'],
                'src_text': sentence['src_text'],
                'mt_text': sentence['mt_text'],
                'ref_text': sentence['ref_text'],
                'mqm_analysis': {"错误": "API调用失败"},
                'raw_response': None
            }
        
        results.append(result)
        
        # 实时保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')
        
        # 避免过于频繁的API调用
        time.sleep(0.5)
        
        # 每100个保存一次进度提示
        if (i + 1) % 100 == 0:
            print(f"已完成 {i + 1} 个句子，结果已保存到 {output_file}")
    
    print(f"处理完成！共处理 {len(results)} 个句子")
    return results

def generate_summary_report(results, output_dir):
    """生成汇总报告"""
    print("生成汇总报告...")
    
    # 统计错误类型分布
    error_type_count = Counter()
    severity_count = Counter()
    confidence_count = Counter()
    
    for result in results:
        mqm_analysis = result.get('mqm_analysis', {})
        
        error_type = mqm_analysis.get('error_category', '未知')
        severity = mqm_analysis.get('severity', '未知')
        confidence = mqm_analysis.get('confidence', '未知')
        
        error_type_count[error_type] += 1
        severity_count[severity] += 1
        confidence_count[confidence] += 1
    
    # 生成报告
    report = {
        '总句子数': len(results),
        '错误类型分布': dict(error_type_count),
        '错误严重程度分布': dict(severity_count),
        '置信度分布': dict(confidence_count)
    }
    
    # 保存报告
    report_file = os.path.join(output_dir, 'mqm_classification_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 打印报告
    print("\n--- MQM错误分类报告 ---")
    print(f"总句子数: {report['总句子数']}")
    
    print("\n错误类型分布:")
    for error_type, count in sorted(error_type_count.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results)) * 100
        print(f"  {error_type}: {count} ({percentage:.1f}%)")
    
    print("\n错误严重程度分布:")
    for severity, count in severity_count.items():
        percentage = (count / len(results)) * 100
        print(f"  {severity}: {count} ({percentage:.1f}%)")
    
    # 不再输出与触发词相关性的统计
    
    print(f"\n报告已保存: {report_file}")

def main():
    """主函数"""
    print("MQM错误分类分析开始...")
    
    # 配置要处理的语言对
    language_pairs = ['zhen3']  # 可以修改为 ['zhen2', 'zhen3', 'aren2', 'aren3']
    
    for lang_pair in language_pairs:
        print(f"\n处理语言对: {lang_pair}")
        
        # 构建句子数据
        sentence_data, src_lang = build_sentence_data('.', lang_pair)
        
        if not sentence_data:
            print(f"跳过 {lang_pair}，数据构建失败")
            continue
        
        # 设置输出路径
        output_dir = f'bad_word/{lang_pair}'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'mqm_classification_results.jsonl')
        
        # 处理句子（支持断点续传）
        resume_from = 0
        if os.path.exists(output_file):
            # 计算已处理的句子数
            with open(output_file, 'r', encoding='utf-8') as f:
                resume_from = sum(1 for _ in f)
            print(f"检测到已有结果文件，从第 {resume_from + 1} 个句子开始")
        
        results = process_sentences(sentence_data, output_file, resume_from)
        
        # 生成汇总报告
        generate_summary_report(results, output_dir)
    
    print("\n所有语言对处理完成！")

if __name__ == "__main__":
    main()
