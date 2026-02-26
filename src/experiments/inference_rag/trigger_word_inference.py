import json
import os
import ast
from src.utils.common import *

def readline(path):
    with open(path, 'r', encoding='utf-8') as f:
        a = [i[:-1] for i in f.readlines()]
    return a

def readlist(path):
    with open(path, 'r') as f:
        content = f.read()
        # 处理可能的格式问题，如果文件内容是类似列表的字符串
        try:
            a = ast.literal_eval(content)
        except:
            # 如果直接读取失败，尝试作为普通json读取，或者处理特定格式
            # 根据描述，comet文件是一个大列表，元素是元组
            # 如果ast.literal_eval失败，可能是因为文件太大或者格式微小差异
            # 这里假设文件内容是合法的python列表字符串
            print(f"Error reading list from {path}")
            return []
    return a

def process_target_bad_words(tgt_path, comet_path):
    print(f"Processing {tgt_path} ...")
    
    # 确定输出目录
    output_dir = os.path.dirname(tgt_path)
    output_all_json = os.path.join(output_dir, 'tgt_ycc_all.json')
    output_key_json = os.path.join(output_dir, 'tgt_ycc_key.json')
    
    # 读取目标端句子
    if not os.path.exists(tgt_path):
        print(f"File not found: {tgt_path}")
        return

    tgt_sentences = readline(tgt_path)
    
    # 读取COMET分数
    if not os.path.exists(comet_path):
        print(f"File not found: {comet_path}")
        return
        
    comet_data = readlist(comet_path)
    
    if len(tgt_sentences) != len(comet_data):
        print(f"Warning: Length mismatch! Sentences: {len(tgt_sentences)}, Scores: {len(comet_data)}")
        # 取较小长度，防止越界
        min_len = min(len(tgt_sentences), len(comet_data))
        tgt_sentences = tgt_sentences[:min_len]
        comet_data = comet_data[:min_len]
    
    # 提取COMET分数
    # 格式兼容：
    # 1. 元组列表: [(9.65, 0.92), ...] -> 取第二个元素
    # 2. 纯分数列表: [0.83, 0.79, ...] -> 直接使用
    comet_scores = []
    if comet_data and len(comet_data) > 0:
        first_item = comet_data[0]
        if isinstance(first_item, (list, tuple)):
            # 情况1: 元组/列表，取第二个元素 (如果长度足够)
            comet_scores = [item[1] if len(item) > 1 else 0.0 for item in comet_data]
        elif isinstance(first_item, (float, int)):
            # 情况2: 纯数字列表
            comet_scores = [float(item) for item in comet_data]
        else:
            print("Warning: Unknown COMET data format. First item:", first_item)
            return
    
    # 计算全局平均分
    all_avg_comet = sum(comet_scores) / len(comet_scores) if comet_scores else 0
    print(f"Global Average COMET: {all_avg_comet}")
    
    # 统计每个词的COMET分数列表
    word_stats = {}
    
    for i, sent in enumerate(tgt_sentences):
        score = comet_scores[i]
        # 空格分词
        words = sent.strip().split()
        
        # 去重，避免一句话中重复词多次统计影响平均分（或者根据需求，这里通常是统计词级贡献，一句话出现多次算一次还是多次？
        # 参考原代码 src_badword.py:
        # for c in src_fc: ... if c in dict ... append
        # 原代码没有去重，这里保持一致，不去重，统计每一次出现。
        
        for word in words:
            # 简单清洗，去掉标点等可能需要，但用户要求“空格分词”，暂时严格按空格
            if word not in word_stats:
                word_stats[word] = []
            word_stats[word].append(score)
            
    # 计算统计指标并排序
    # 排序规则：按平均分降序（参考原代码）
    # 但原代码 src_cfc_list=sorted(src_cfc_dict.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)
    # 实际上我们找易错词，应该关注低分？原代码似乎是输出所有，然后筛选 score < -5
    
    results = []
    for word, scores in word_stats.items():
        count = len(scores)
        avg = sum(scores) / count
        # score 计算公式修正：
        # 用户描述: score=num*(该语向所有句子的平均comet分数-avg_comet)
        # 注意：这通常表示该词导致的“分数下降总和”。如果 avg < all_avg，则 score 为正？
        # 等等，看原代码： d['score'] = (sum(value)/len(value) - all_avg_comet) * len(value)
        # 如果 avg < all_avg，则 score 为负数。
        # 用户描述说筛选 score < -5，说明 score 越负越差。
        # 所以公式应该是: score = (avg - all_avg) * count
        
        score_val = (avg - all_avg_comet) * count
        
        results.append({
            'trigger_word': word,
            'avg_comet': avg,
            'num': count,
            'score': score_val
        })
        
    # 按平均分排序（或者按score排序？）
    # 原代码按 avg_comet 降序
    results.sort(key=lambda x: x['avg_comet'], reverse=True)
    
    print(f"Total unique words: {len(results)}")
    
    # 写入所有词的统计
    with open(output_all_json, 'w', encoding='utf-8') as f:
        for item in results:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
            
    # 筛选易错词
    # 阈值：score < -5 且 avg_comet < all_avg_comet - 0.05
    count_key = 0
    with open(output_key_json, 'w', encoding='utf-8') as f:
        for item in results:
            if item['score'] < -5 and item['avg_comet'] < (all_avg_comet - 0.05):
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
                count_key += 1
                
    print(f"Identified {count_key} error-prone words. Saved to {output_key_json}")
    print("-" * 50)

def main():
    tasks = [
        # zhen3
        {
            'tgt': '/public/home/xiangyuduan/lyt/bad_word/key_data/zhen3/119w.en',
            'comet': '/public/home/xiangyuduan/lyt/bad_word/key_data/zhen3/119w.comet'
        },
        # zhen2
        {
            'tgt': '/public/home/xiangyuduan/lyt/bad_word/key_data/zhen2/119w.en',
            'comet': '/public/home/xiangyuduan/lyt/bad_word/key_data/zhen2/119w.comet'
        },
        # ende3
        {
            'tgt': '/public/home/xiangyuduan/lyt/bad_word/key_data/ende3/120w.de',
            'comet': '/public/home/xiangyuduan/lyt/bad_word/key_data/ende3/120w.comet'
        },
        # ende2
        {
            'tgt': '/public/home/xiangyuduan/lyt/bad_word/key_data/ende2/120w.de',
            'comet': '/public/home/xiangyuduan/lyt/bad_word/key_data/ende2/120w.comet'
        },
        # aren3
        {
            'tgt': '/public/home/xiangyuduan/lyt/bad_word/key_data/aren3/99w.en',
            'comet': '/public/home/xiangyuduan/lyt/bad_word/key_data/aren3/99w.comet'
        },
        # aren2
        {
            'tgt': '/public/home/xiangyuduan/lyt/bad_word/key_data/aren2/99w.en',
            'comet': '/public/home/xiangyuduan/lyt/bad_word/key_data/aren2/99w.comet'
        }
    ]
    
    for task in tasks:
        process_target_bad_words(task['tgt'], task['comet'])

if __name__ == '__main__':
    main()
