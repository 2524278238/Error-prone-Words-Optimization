#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import argparse
import numpy as np
from pathlib import Path
from src.utils.common import *

def parse_args():
    parser = argparse.ArgumentParser(description="综合comet分数和目标长度采样训练数据")
    parser.add_argument("--num", type=int, default=15000, help="采样数量")
    parser.add_argument("--src_file", type=str, default="/public/home/xiangyuduan/lyt/basedata/125/train.ch", 
                        help="源语言文件路径")
    parser.add_argument("--tgt_file", type=str, default="/public/home/xiangyuduan/lyt/basedata/125/train.en", 
                        help="目标语言文件路径")
    parser.add_argument("--output_dir", type=str, default="/public/home/xiangyuduan/lyt/bad_word/train/train_data",
                        help="输出目录")
    parser.add_argument("--alpha", type=float, default=0.8, help="comet分数权重")
    parser.add_argument("--beta", type=float, default=0.2, help="长度权重")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
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

def cometlen_sample(src_lines, tgt_lines, n, alpha=0.5, beta=0.5):
    print("批量计算COMET分数...")
    comet_scores=readlist('/public/home/xiangyuduan/lyt/bad_word/key_data/119w.cometfree')
    tgt_lens = [len(t.split()) for t in tgt_lines]
    max_len = max(tgt_lens)
    norm_lens = [l / max_len for l in tgt_lens]
    # 综合分数
    scores = [alpha * c + beta * l for c, l in zip(comet_scores, norm_lens)]
    indexed_scores = [(i, s) for i, s in enumerate(scores)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
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
    print(f"综合采样 {args.num} 个样本...")
    sampled_src, sampled_tgt, indices = cometlen_sample(src_lines, tgt_lines, args.num, args.alpha, args.beta)
    # 英文句子格式处理
    sampled_tgt = correct_en_lines(sampled_tgt)
    # 输出
    src_output = os.path.join(args.output_dir, f"train_src_cometlen.zh")
    tgt_output = os.path.join(args.output_dir, f"train_ref_cometlen.en")
    indices_output = os.path.join(args.output_dir, f"train_indices_cometlen.json")
    print(f"保存源语言采样结果到: {src_output}")
    with open(src_output, "w", encoding="utf-8") as f:
        f.write("\n".join(sampled_src))
    print(f"保存目标语言采样结果到: {tgt_output}")
    with open(tgt_output, "w", encoding="utf-8") as f:
        for line in sampled_tgt:
            f.write(line + '\n')
    print(f"保存采样索引到: {indices_output}")
    with open(indices_output, "w", encoding="utf-8") as f:
        json.dump(indices, f)
    print(f"采样完成，总共采样了 {len(sampled_src)} 个样本")

if __name__ == "__main__":
    main() 