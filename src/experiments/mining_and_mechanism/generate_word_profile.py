#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合易错词分析结果，生成易错词档案
规定字段：
- word: 易错词
- err: 错翻率
- oms: 漏翻率
- pos: 词性
- type: 主错误类型
- type_dict: 错误类型分布
"""

import json
import os
from collections import defaultdict, Counter

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_error_rates_from_alignment_analysis(base_dir):
    """从alignment_based_analysis_results.json获取错翻率和漏翻率"""
    alignment_file = os.path.join(base_dir, "alignment_based_analysis_results.json")
    
    if not os.path.exists(alignment_file):
        print(f"警告: {alignment_file} 不存在")
        return {}
    
    alignment_data = load_json(alignment_file)
    
    word_rates = {}
    
    # 遍历所有触发词的分析结果
    for word, word_data in alignment_data.items():
        # 使用文件中的统计数据
        total_sentences = word_data.get('total_sentences', 0)
        mistranslated = word_data.get('mistranslated', 0)
        omitted = word_data.get('omitted', 0)
        
        if total_sentences == 0:
            continue
            
        # 计算比率
        err_rate = mistranslated / total_sentences
        oms_rate = omitted / total_sentences
        
        word_rates[word] = {
            'err': round(err_rate, 4),
            'oms': round(oms_rate, 4),
            'total_samples': total_sentences
        }
    
    return word_rates

def get_pos_tags(pos_file):
    """从词性文件获取词性标注"""
    if not os.path.exists(pos_file):
        print(f"警告: {pos_file} 不存在")
        return {}
    
    pos_data = load_jsonl(pos_file)
    
    word_pos = {}
    for item in pos_data:
        word = item.get('word', '')
        category = item.get('category', '')
        
        word_pos[word] = category
    
    return word_pos

def get_error_type_distribution(mqm_file):
    """从MQM分类结果获取错误类型分布"""
    if not os.path.exists(mqm_file):
        print(f"警告: {mqm_file} 不存在")
        return {}
    
    mqm_data = load_jsonl(mqm_file)
    
    # 按易错词分组统计错误类型
    word_errors = defaultdict(list)
    
    for item in mqm_data:
        trigger_word = item.get('trigger_word', '')
        mqm_analysis = item.get('mqm_analysis', {})
        error_category = mqm_analysis.get('error_category', '')
        
        if trigger_word and error_category:
            word_errors[trigger_word].append(error_category)
    
    # 计算每个词的错误类型分布
    word_type_dist = {}
    
    for word, errors in word_errors.items():
        error_counter = Counter(errors)
        total = len(errors)
        
        # 错误类型分布（比例）
        type_dict = {error_type: round(count/total, 4) 
                    for error_type, count in error_counter.items()}
        
        # 主错误类型（最常见的）
        main_type = error_counter.most_common(1)[0][0] if error_counter else ""
        
        word_type_dist[word] = {
            'type': main_type,
            'type_dict': type_dict,
            'total_samples': total
        }
    
    return word_type_dist

def generate_word_profiles(base_dir, output_file):
    """生成易错词档案"""
    
    # 1. 获取错翻率和漏翻率
    print("加载错翻率和漏翻率数据...")
    error_rates = get_error_rates_from_alignment_analysis(base_dir)
    
    # 2. 获取词性标注
    pos_file = os.path.join(base_dir, "zhen2_very_pos_tags_fixed.jsonl")
    print("加载词性标注数据...")
    pos_tags = get_pos_tags(pos_file)
    
    # 3. 获取错误类型分布
    mqm_file = os.path.join(base_dir, "mqm_classification_results.jsonl")
    print("加载错误类型分布数据...")
    error_types = get_error_type_distribution(mqm_file)
    
    # 4. 整合数据
    print("整合数据...")
    
    # 获取所有词的集合
    all_words = set()
    all_words.update(error_rates.keys())
    all_words.update(pos_tags.keys())
    all_words.update(error_types.keys())
    
    profiles = []
    
    for word in all_words:
        profile = {
            'word': word,
            'err': error_rates.get(word, {}).get('err', 0.0),
            'oms': error_rates.get(word, {}).get('oms', 0.0),
            'pos': pos_tags.get(word, '未知'),
            'type': error_types.get(word, {}).get('type', ''),
            'type_dict': error_types.get(word, {}).get('type_dict', {}),
            # 额外信息用于分析
            'alignment_samples': error_rates.get(word, {}).get('total_samples', 0),
            'mqm_samples': error_types.get(word, {}).get('total_samples', 0)
        }
        profiles.append(profile)
    
    # 5. 按样本数量排序（优先显示有更多样本的词）
    profiles.sort(key=lambda x: x['mqm_samples'] + x['alignment_samples'], reverse=True)
    
    # 6. 保存结果
    print(f"保存结果到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for profile in profiles:
            f.write(json.dumps(profile, ensure_ascii=False) + '\n')
    
    # 7. 生成统计报告
    print("\n=== 易错词档案统计报告 ===")
    print(f"总词汇数: {len(profiles)}")
    
    # 有对齐分析数据的词数
    with_alignment = [p for p in profiles if p['alignment_samples'] > 0]
    print(f"有对齐分析数据的词数: {len(with_alignment)}")
    
    # 有词性标注的词数
    with_pos = [p for p in profiles if p['pos'] != '未知']
    print(f"有词性标注的词数: {len(with_pos)}")
    
    # 有错误类型分析的词数
    with_error_type = [p for p in profiles if p['type']]
    print(f"有错误类型分析的词数: {len(with_error_type)}")
    
    # 完整数据的词数
    complete_data = [p for p in profiles 
                    if p['alignment_samples'] > 0 and p['pos'] != '未知' and p['type']]
    print(f"有完整数据的词数: {len(complete_data)}")
    
    # 错翻率和漏翻率统计
    if with_alignment:
        err_rates = [p['err'] for p in with_alignment]
        oms_rates = [p['oms'] for p in with_alignment]
        
        print(f"\n错翻率统计:")
        print(f"  平均: {sum(err_rates)/len(err_rates):.4f}")
        print(f"  中位数: {sorted(err_rates)[len(err_rates)//2]:.4f}")
        print(f"  最大: {max(err_rates):.4f}")
        
        print(f"\n漏翻率统计:")
        print(f"  平均: {sum(oms_rates)/len(oms_rates):.4f}")
        print(f"  中位数: {sorted(oms_rates)[len(oms_rates)//2]:.4f}")
        print(f"  最大: {max(oms_rates):.4f}")
    
    # 词性分布
    if with_pos:
        pos_counter = Counter([p['pos'] for p in with_pos])
        print(f"\n词性分布 (Top 10):")
        for pos, count in pos_counter.most_common(10):
            print(f"  {pos}: {count}")
    
    # 主错误类型分布
    if with_error_type:
        type_counter = Counter([p['type'] for p in with_error_type])
        print(f"\n主错误类型分布 (Top 10):")
        for error_type, count in type_counter.most_common(10):
            print(f"  {error_type}: {count}")
    
    return profiles

def main():
    """主函数"""
    # 配置文件路径
    base_dir = "bad_word/zhen2"  # 数据目录
    output_file = "bad_word/zhen2/word_profiles.jsonl"  # 输出文件
    
    # 检查目录是否存在
    if not os.path.exists(base_dir):
        print(f"错误: 目录 {base_dir} 不存在")
        return
    
    # 生成易错词档案
    profiles = generate_word_profiles(base_dir, output_file)
    
    print(f"\n易错词档案已生成: {output_file}")
    print(f"总计 {len(profiles)} 个词汇的档案")

if __name__ == "__main__":
    main()
