#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from src.utils.common import readall, readline, count_comet,readlist
from datetime import datetime
from rank_bm25 import BM25Okapi
import jieba
import time

def parse_args():
    parser = argparse.ArgumentParser(description="采样训练数据（引入长度权重）")
    parser.add_argument("--mode", type=str, required=True, choices=["random", "comet", "trigger", "ppl"],
                        help="采样模式: random(随机), comet(高分), trigger(触发词), ppl(低困惑度)")
    parser.add_argument("--num", type=int, default=15000, help="采样数量")
    parser.add_argument("--src_file", type=str, default="/public/home/xiangyuduan/lyt/basedata/125/train.ch", 
                        help="源语言文件路径")
    parser.add_argument("--tgt_file", type=str, default="/public/home/xiangyuduan/lyt/basedata/125/train.en", 
                        help="目标语言文件路径")
    parser.add_argument("--trigger_json", type=str, default="/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/key_with_sent_all.json",
                        help="触发词信息JSON文件路径")
    parser.add_argument("--output_dir", type=str, default="/public/home/xiangyuduan/lyt/bad_word/train/train_data",
                        help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--ppl_threshold", type=float, default=100.0, help="困惑度阈值，超过该值的句子不采样 (仅trigger模式)")
    return parser.parse_args()

def correct_en_lines(lines):
    # 按correct.py规则处理英文句子
    new_lines = []
    for line in lines:
        line = line.replace('$ ', '$')
        line = line.replace(' :', ':')
        line = line.replace(" n't", "n't")
        line = line.replace('[ ', '[')
        line = line.replace(' ]', ']')
        line = line.replace(' / ', '/')
        line = line.replace(' ;', ';')
        line = line.replace(',', ', ')
        line = line.replace(',  ', ', ')
        line = line.replace('- - -', '---')
        line = line.replace(" 's", "'s")
        line = line.replace('(', ' (')
        line = line.replace(')', ') ').strip()
        new_lines.append(line)
    return new_lines

def random_sample(src_lines, tgt_lines, n, seed=42):
    random.seed(seed)
    total = len(src_lines)
    indices = random.sample(range(total), n)
    return [src_lines[i] for i in indices], [tgt_lines[i] for i in indices], indices

def comet_sample(src_lines, tgt_lines, n):
    print("计算所有样本的COMET分数...")
    comet_scores, _ = count_comet(src_lines, tgt_lines, tgt_lines, model_path='/public/home/xiangyuduan/bli/blidata/models/hf/wmt22-cometkiwi-da/checkpoints/model.ckpt')
    with open('/public/home/xiangyuduan/lyt/bad_word/key_data/119w.cometfree','w',encoding="utf-8")as f:
        f.write(str(comet_scores))
    indexed_scores = [(i, score) for i, score in enumerate(comet_scores)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in indexed_scores[:n]]
    return [src_lines[i] for i in top_indices], [tgt_lines[i] for i in top_indices], top_indices

def trigger_sample(src_lines, tgt_lines, trigger_data, n, ppl_threshold=100.0):
    """基于触发词采样n个样本，综合COMET分数和长度权重，并过滤高困惑度，且用BM25全局去重相似句，支持进度和剩余时间实时打印"""
    comet=readlist('/public/home/xiangyuduan/lyt/bad_word/key_data/119w.cometfree')
    all_ppl = readlist('/public/home/xiangyuduan/lyt/bad_word/key_data/119w.ppl')
    trigger_weights = {}
    total_weight = 0
    for item in trigger_data:
        trigger_word = item["trigger_word"]
        num = item["num"]
        avg_comet = item["avg_comet"]
        score = abs(item["score"])
        weight = num * (1 - avg_comet) / (score + 1e-10)
        #weight = np.log(score)
        trigger_weights[trigger_word] = weight
        total_weight += weight
    trigger_samples = {}
    remaining = n
    assigned = 0
    max_sample_count = 20
    
    for trigger_word, weight in sorted(trigger_weights.items(), key=lambda x: x[1], reverse=True):
        sample_count = int(n * weight / total_weight)
        sample_count = max(1, sample_count)
        trigger_info = next(item for item in trigger_data if item["trigger_word"] == trigger_word)
        available = len(trigger_info["index_list"])
        sample_count = min(sample_count, max_sample_count)
        sample_count = min(sample_count, remaining)
        trigger_samples[trigger_word] = sample_count
        assigned += sample_count
        remaining = n - assigned
        if remaining <= 0:
            break
    if remaining > 0:
        for trigger_word, weight in sorted(trigger_weights.items(), key=lambda x: x[1], reverse=True):
            if trigger_word not in trigger_samples:
                continue
            trigger_info = next(item for item in trigger_data if item["trigger_word"] == trigger_word)
            available = len(trigger_info["index_list"]) - trigger_samples[trigger_word]
            extra = min(remaining, max_sample_count)
            trigger_samples[trigger_word] += extra
            remaining -= extra
            if remaining <= 0:
                break
    global_selected_indices = []
    global_selected_texts = []
    # print("每个触发词采样的数据条数:")
    # print([count for trigger_word, count in trigger_samples.items()])
    last_report = 0
    start_time = time.time()

    for trigger_word, count in trigger_samples.items():
        if count <= 0:
            continue
        trigger_info = next(item for item in trigger_data if item["trigger_word"] == trigger_word)
        indices_with_scores = trigger_info["index_list"]
        # 过滤掉ppl超阈值的句子
        filtered_indices = [idx for idx in indices_with_scores if all_ppl[idx] <= ppl_threshold]
        if not filtered_indices:
            continue
        lengths = [len(src_lines[idx]) for idx in filtered_indices]
        comet_scores = [comet[i] for i in filtered_indices]
        ppl_scores = [all_ppl[i] for i in filtered_indices]
        def normalize(arr):
            arr = np.array(arr)
            if arr.max() == arr.min():
                return np.zeros_like(arr)
            return (arr - arr.min()) / (arr.max() - arr.min())
        comet_norm = normalize(comet_scores)
        length_norm = normalize(lengths)
        ppl_norm = normalize(ppl_scores) 
        alpha, beta, gamma = 1, 0, 0
        combined = comet_norm * alpha + length_norm * beta + ppl_norm * gamma
        sorted_indices = np.argsort(combined)[::-1]
        for i in sorted_indices:
            idx = filtered_indices[i]
            candidate = src_lines[idx]
            candidate_tokens = candidate.split()
            # 构建全局BM25语料库
            # bm25_corpus = [text.split() for text in global_selected_texts] + [candidate_tokens]
            # bm25 = BM25Okapi(bm25_corpus)
            # is_similar = False
            # if global_selected_texts:
            #     scores = bm25.get_scores(candidate_tokens)
            #     for sel_pos in range(len(global_selected_texts)):
            #         if scores[sel_pos] > 5:  # 建议阈值5
            #             is_similar = True
            #             break
            # if is_similar:
            #     continue
            global_selected_indices.append(idx)
            global_selected_texts.append(candidate)
            if len(global_selected_indices) >= n:
                break
        if len(global_selected_indices) >= n:
            break
    global_selected_indices = global_selected_indices[:n]
    return [src_lines[i] for i in global_selected_indices], [tgt_lines[i] for i in global_selected_indices], global_selected_indices

def compute_ppl(sentences, model_name="/public/home/xiangyuduan/bli/blidata/models/hf/Qwen2.5-3B"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import numpy as np
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
    ppls = []
    total = len(sentences)
    for idx, sent in enumerate(sentences):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - processing sentence {idx+1}/{total}")
        enc = tokenizer(sent, return_tensors="pt")
        input_ids = enc["input_ids"].cuda()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss.item()
        ppl = np.exp(loss)
        ppls.append(ppl)
    with open('/public/home/xiangyuduan/lyt/bad_word/key_data/119w.ppl','w',encoding="utf-8")as f:
        f.write(str(ppls))
    return ppls

def ppl_sample(src_lines, tgt_lines, n, model_name="/public/home/xiangyuduan/bli/blidata/models/hf/Qwen2.5-3B"):
    print("计算所有样本的困惑度...")
    ppls = compute_ppl(src_lines, model_name)
    indexed_scores = [(i, score) for i, score in enumerate(ppls)]
    indexed_scores.sort(key=lambda x: x[1])  # 困惑度越低越好
    top_indices = [idx for idx, _ in indexed_scores[:n]]
    return [src_lines[i] for i in top_indices], [tgt_lines[i] for i in top_indices], top_indices

def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"加载源语言文件: {args.src_file}")
    src_lines = readline(args.src_file)
    print(f"加载目标语言文件: {args.tgt_file}")
    tgt_lines = readline(args.tgt_file)
    assert len(src_lines) == len(tgt_lines), "源语言和目标语言文件行数不匹配！"
    print(f"总行数: {len(src_lines)}")
    if args.mode == "random":
        print(f"随机采样 {args.num} 个样本...")
        sampled_src, sampled_tgt, indices = random_sample(src_lines, tgt_lines, args.num, args.seed)
        output_suffix = "random"
    elif args.mode == "comet":
        print(f"基于COMET评分采样 {args.num} 个高分样本...")
        sampled_src, sampled_tgt, indices = comet_sample(src_lines, tgt_lines, args.num)
        output_suffix = "comet"
    elif args.mode == "trigger":
        print(f"加载触发词信息: {args.trigger_json}")
        with open(args.trigger_json, "r", encoding="utf-8") as f:
            trigger_data = [json.loads(line) for line in f.readlines()]
        print(f"基于触发词采样 {args.num} 个样本...")
        sampled_src, sampled_tgt, indices = trigger_sample(src_lines, tgt_lines, trigger_data, args.num, args.ppl_threshold)
        output_suffix = "triggerlen"
    elif args.mode == "ppl":
        print(f"基于困惑度采样 {args.num} 个低困惑度样本...")
        sampled_src, sampled_tgt, indices = ppl_sample(src_lines, tgt_lines, args.num)
        output_suffix = "ppl"

    sampled_tgt = correct_en_lines(sampled_tgt)
    src_output = os.path.join(args.output_dir, f"train_src_{output_suffix}.zh")
    tgt_output = os.path.join(args.output_dir, f"train_ref_{output_suffix}.en")
    indices_output = os.path.join(args.output_dir, f"train_indices_{output_suffix}.json")
    print(f"保存源语言采样结果到: {src_output}")
    with open(src_output, "w", encoding="utf-8") as f:
        f.write("\n".join(sampled_src))
    print(f"保存目标语言采样结果到: {tgt_output}")
    with open(tgt_output, "w", encoding="utf-8") as f:
        f.write("\n".join(sampled_tgt))
    print(f"保存采样索引到: {indices_output}")
    with open(indices_output, "w", encoding="utf-8") as f:
        json.dump(indices, f)
    print(f"采样完成，总共采样了 {len(sampled_src)} 个样本")

if __name__ == "__main__":
    main() 