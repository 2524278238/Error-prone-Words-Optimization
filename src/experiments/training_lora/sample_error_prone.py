#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import argparse
import numpy as np
from src.utils.common import readall, readline, count_comet, readlist
# 在导入 torch 之前设置环境变量以减少内存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

# 路径映射配置
CONFIG = {
    'aren': {
        'train_src': '/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.ar',
        'train_ref': '/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.en',
        'ppl_file': '/public/home/xiangyuduan/lyt/bad_word/key_data/aren/train.ppl', 
        'tgt_is_en': True
    },
    'zhen': {
        'train_src': '/public/home/xiangyuduan/lyt/basedata/125/train.ch',
        'train_ref': '/public/home/xiangyuduan/lyt/basedata/125/train.en',
        'ppl_file': '/public/home/xiangyuduan/lyt/bad_word/key_data/zhen/train.ppl',
        'tgt_is_en': True
    },
    'ende': {
        'train_src': '/public/home/xiangyuduan/lyt/basedata/459/train.en',
        'train_ref': '/public/home/xiangyuduan/lyt/basedata/459/train.de',
        'ppl_file': '/public/home/xiangyuduan/lyt/bad_word/key_data/ende/train.ppl',
        'tgt_is_en': False
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="基于目标端易错词采样训练数据")
    parser.add_argument("--lang", type=str, required=True, choices=["aren", "zhen", "ende"], help="语向")
    parser.add_argument("--model_ver", type=str, default="llama3", choices=["llama2", "llama3"], help="模型版本")
    parser.add_argument("--num", type=int, default=17300, help="采样数量")
    parser.add_argument("--output_dir", type=str, default="/public/home/xiangyuduan/lyt/bad_word/train/train_data", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    return parser.parse_args()

def compute_ppl(sentences, output_path, model_name="/public/home/xiangyuduan/lyt/model/Llama-3.2-3B"):
    if os.path.exists(output_path):
        print(f"Loading existing PPL from {output_path}...")
        try:
            return readlist(output_path)
        except Exception as e:
            print(f"Failed to load PPL: {e}, re-computing...")
    
    print(f"Computing PPL for {len(sentences)} sentences...")
    
    # 清理内存
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
        model.eval()
    except Exception as e:
        print(f"Error loading model for PPL: {e}")
        return None
    
    ppls = []
    # 初始 batch size 降低为 8，更加保守
    initial_batch_size = 32
    total = len(sentences)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    
    i = 0
    while i < total:
        if i % 1000 == 0:
            print(f"Processing {i}/{total}...")
            # 定期清理缓存
            torch.cuda.empty_cache()
            
        # 动态尝试 batch size
        current_batch_size = initial_batch_size
        success = False
        
        while not success and current_batch_size >= 1:
            try:
                batch_sents = sentences[i:min(i+current_batch_size, total)]
                
                # 注意：max_length 设置限制，防止极长句子爆显存
                inputs = tokenizer(batch_sents, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
                
                with torch.no_grad():
                    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                    logits = outputs.logits
                    
                    # Shift
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs["input_ids"][..., 1:].contiguous()
                    shift_mask = inputs["attention_mask"][..., 1:].contiguous()
                    
                    # Calculate loss per token
                    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                    flat_labels = shift_labels.view(-1)
                    loss = loss_fct(flat_logits, flat_labels)
                    
                    # Reshape
                    loss = loss.view(shift_labels.size())
                    
                    # Mask padding
                    loss = loss * shift_mask
                    
                    # Average loss per sentence
                    seq_lens = shift_mask.sum(dim=1)
                    per_sent_loss = loss.sum(dim=1) / (seq_lens + 1e-10)
                    
                    batch_ppls = torch.exp(per_sent_loss).cpu().numpy().tolist()
                    ppls.extend(batch_ppls)
                    
                success = True
                i += current_batch_size
                
            except torch.cuda.OutOfMemoryError:
                print(f"OOM at index {i} with batch_size {current_batch_size}. Retrying with smaller batch...")
                torch.cuda.empty_cache()
                current_batch_size //= 2
                if current_batch_size < 1:
                    print(f"Skipping index {i} due to OOM even with batch_size 1.")
                    # 如果单个句子都 OOM，填入一个较大的 PPL 值（如 10000）表示质量差/不可计算
                    ppls.extend([10000.0] * (min(i+initial_batch_size, total) - i)) 
                    i += initial_batch_size # 跳过这一段
                    success = True # 强制继续
            except Exception as e:
                print(f"Error at index {i}: {e}")
                # 出错也跳过
                ppls.extend([10000.0] * (min(i+current_batch_size, total) - i))
                i += current_batch_size
                success = True

    # 释放模型
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
        
    # Save PPL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding="utf-8") as f:
        f.write(str(ppls))
    
    return ppls

def normalize(arr):
    arr = np.array(arr)
    if len(arr) == 0:
        return arr
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())

def trigger_sample(src_lines, tgt_lines, trigger_data, n, ppl_scores=None):
    """
    基于目标端易错词采样
    权重计算: num * (1 - avg_comet) / score
    候选筛选: COMET高分, PPL低分
    """
    
    trigger_weights = {}
    total_weight = 0
    
    # 1. 计算权重
    for item in trigger_data:
        trigger_word = item["trigger_word"]
        num = item["num"]
        avg_comet = item["avg_comet"]
        score = abs(item["score"]) # score is negative
        
        # 目标端触发词权重计算公式
        weight = num * (1 - avg_comet) / (score + 1e-10)
        trigger_weights[trigger_word] = weight
        total_weight += weight
        
    print(f"Total trigger words: {len(trigger_weights)}")
    
    # 2. 分配样本数量
    trigger_samples = {}
    remaining = n
    assigned = 0
    
    # 按权重降序分配
    sorted_triggers = sorted(trigger_weights.items(), key=lambda x: x[1], reverse=True)
    
    for trigger_word, weight in sorted_triggers:
        if total_weight > 0:
            sample_count = int(n * weight / total_weight)
        else:
            sample_count = 0
            
        # 至少分配1个
        sample_count = max(1, sample_count)
        
        # 查找可用样本数
        trigger_info = next(item for item in trigger_data if item["trigger_word"] == trigger_word)
        available = len(trigger_info["index_list"])
        
        # 限制：不要超过可用样本
        sample_count = min(sample_count, available)
        
        # 限制：不要超过剩余总配额
        sample_count = min(sample_count, remaining)
        
        trigger_samples[trigger_word] = sample_count
        assigned += sample_count
        remaining = n - assigned
        
        if remaining <= 0:
            break
            
    # 如果还有剩余，继续分配给高权重词
    if remaining > 0:
        for trigger_word, weight in sorted_triggers:
            if trigger_word not in trigger_samples:
                continue
            
            trigger_info = next(item for item in trigger_data if item["trigger_word"] == trigger_word)
            available = len(trigger_info["index_list"]) - trigger_samples[trigger_word]
            
            extra = min(remaining, available)
            trigger_samples[trigger_word] += extra
            remaining -= extra
            
            if remaining <= 0:
                break
                
    print(f"Assigned {assigned + (n-remaining)} samples across {len(trigger_samples)} triggers.")

    # 3. 选择具体的样本
    selected_indices = []
    
    for trigger_word, count in trigger_samples.items():
        if count <= 0:
            continue
            
        trigger_info = next(item for item in trigger_data if item["trigger_word"] == trigger_word)
        
        # index_list 已经是 [[idx, score], ...]
        # 或者是 [idx, ...] (如果 generate_tgt_comet_index.py 没跑对的话，但我们假设它跑对了)
        # 我们需要兼容一下
        raw_indices = trigger_info["index_list"]
        if not raw_indices:
            continue
            
        # 标准化为 list of (idx, comet_score)
        indices_with_scores = []
        if isinstance(raw_indices[0], list):
            indices_with_scores = [(x[0], x[1]) for x in raw_indices]
        else:
            # 如果没有分数，只能盲选，或者假定0.5
            indices_with_scores = [(x, 0.5) for x in raw_indices]
            
        # 过滤和打分
        # 如果有 PPL，结合 PPL
        if ppl_scores:
            # 过滤逻辑: PPL <= 150 且 COMET > 0.6
            
            filtered = [(i, c) for i, c in indices_with_scores if ppl_scores[i] <= 150 and c > 0.6]
            if len(filtered) < count:
                    # 放宽条件
                    filtered = indices_with_scores
            
            if not filtered:
                continue
                
            # 综合打分
            c_scores = [c for _, c in filtered]
            p_scores = [ppl_scores[i] for i, _ in filtered]
            l_scores = [len(tgt_lines[i]) for i, _ in filtered] # 使用 Target 长度
            
            c_norm = normalize(c_scores)
            p_norm = normalize(p_scores)
            l_norm = normalize(l_scores)
            
            # 组合分数：COMET 越高越好，PPL 越低越好
            combined = 0.8 * c_norm + 0.2 * (1 - p_norm)
        
            # 重新打包
            candidates = [(filtered[k][0], combined[k]) for k in range(len(filtered))]
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            selected = [x[0] for x in candidates[:count]]
            selected_indices.extend(selected)
            
        else:
            # 只有 COMET
            # 过滤
            filtered = [x for x in indices_with_scores if x[1] > 0.6]
            if len(filtered) < count:
                filtered = indices_with_scores
                
            # 排序
            filtered.sort(key=lambda x: x[1], reverse=True)
            selected = [x[0] for x in filtered[:count]]
            selected_indices.extend(selected)

    # 截断
    selected_indices = selected_indices[:n]
    
    return [src_lines[i] for i in selected_indices], [tgt_lines[i] for i in selected_indices], selected_indices

def correct_en_lines(lines):
    # 简单的英文纠错，保留原逻辑
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

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    
    # 路径配置
    cfg = CONFIG.get(args.lang)
    if not cfg:
        print(f"Error: Unknown language {args.lang}")
        sys.exit(1)
        
    print(f"Processing {args.lang} with model version {args.model_ver}")
    
    # 1. 加载数据
    print(f"Loading src: {cfg['train_src']}")
    src_lines = readline(cfg['train_src'])
    print(f"Loading ref: {cfg['train_ref']}")
    tgt_lines = readline(cfg['train_ref'])
    
    assert len(src_lines) == len(tgt_lines), "Source and Target lines mismatch"
    
    # 2. 加载 Trigger Data
    data_dir_name = f"{args.lang}{'2' if args.model_ver == 'llama2' else '3'}"
    trigger_path = f"/public/home/xiangyuduan/lyt/bad_word/key_data/{data_dir_name}/tgt_key_with_sent_comet.json"
    
    print(f"Loading trigger data: {trigger_path}")
    if not os.path.exists(trigger_path):
        print("Trigger file not found!")
        sys.exit(1)
        
    with open(trigger_path, 'r', encoding='utf-8') as f:
        trigger_data = [json.loads(line) for line in f]
        
    # 3. 加载/计算 PPL
    # PPL 是基于 Target Sentence 计算的 (衡量 Target 的流畅度/常见度)
    ppl_file = cfg.get('ppl_file')
    ppl_scores = None
    
    if ppl_file:
        # 检查是否可以计算 (如果文件不存在)
        if not os.path.exists(ppl_file):
            print("PPL file not found. Attempting to compute (this may take a while)...")
            try:
                ppl_scores = compute_ppl(tgt_lines, ppl_file)
            except Exception as e:
                print(f"PPL computation failed: {e}. Proceeding without PPL.")
        else:
             ppl_scores = readlist(ppl_file)
    
    if ppl_scores and len(ppl_scores) != len(tgt_lines):
        print("Warning: PPL scores length mismatch. Ignoring PPL.")
        ppl_scores = None
        
    # 4. 采样
    print(f"Sampling {args.num} examples...")
    sampled_src, sampled_tgt, indices = trigger_sample(src_lines, tgt_lines, trigger_data, args.num, ppl_scores)
    
    # 5. 后处理 (如果是英文目标端)
    if cfg['tgt_is_en']:
        print("Applying English correction...")
        sampled_tgt = correct_en_lines(sampled_tgt)
        
    # 6. 保存
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"tgt_trigger_{args.lang}_{args.model_ver}"
    
    out_src = os.path.join(args.output_dir, f"train_src_{suffix}.src")
    out_tgt = os.path.join(args.output_dir, f"train_ref_{suffix}.tgt") # .tgt 方便区分
    out_idx = os.path.join(args.output_dir, f"train_indices_{suffix}.json")
    
    print(f"Saving to {out_src} and {out_tgt}")
    with open(out_src, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sampled_src))
    with open(out_tgt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sampled_tgt))
    with open(out_idx, 'w', encoding='utf-8') as f:
        json.dump(indices, f)
        
    print("Done.")

if __name__ == "__main__":
    main()
