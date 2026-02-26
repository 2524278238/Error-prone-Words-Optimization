#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用awesome-align找到触发词在翻译句对中的词对齐 (支持中英和阿英, 使用pyarabic)
"""

import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path
import shutil

# 根据语言选择性导入
try:
    import jieba
except ImportError:
    jieba = None

try:
    import pyarabic.araby as araby
except ImportError:
    araby = None


class WordAlignmentFinder:
    def __init__(self, awesome_align_path, model_path, lang='zh'):
        """
        初始化词对齐查找器。
        
        Args:
            awesome_align_path: awesome-align项目路径
            model_path: 预训练模型路径
            lang: 语言 ('zh' 或 'ar')
        """
        self.awesome_align_path = awesome_align_path
        self.model_path = model_path
        self.lang = lang
        self.tokenizer = self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        """根据语言初始化分词器"""
        if self.lang == 'zh':
            if jieba is None:
                raise ImportError("Jieba not installed. Please run 'pip install jieba'")
            print("Jieba a-a-a...正在努力加载中...")
            jieba.initialize()
            print("Jieba 已准备就绪!")
            return lambda s: jieba.cut(s)
        elif self.lang == 'ar':
            if araby is None:
                raise ImportError("PyArabic not installed. Please run 'pip install pyarabic'")
            print("PyArabic a-a-a...正在努力加载中...")
            # araby.tokenize 返回一个已经分好的词的列表
            print("PyArabic 已准备就绪!")
            return lambda s: araby.tokenize(s)
        else:
            raise ValueError(f"不支持的语言: {self.lang}")

    def _create_temp_input_file(self, src_sentences, tgt_sentences):
        """创建单个临时输入文件，使用指定语言的分词器"""
        temp_dir = '/data/lyt/badword/tmp'
        os.makedirs(temp_dir, exist_ok=True)
        input_file = os.path.join(temp_dir, 'input.txt')
        
        with open(input_file, 'w', encoding='utf-8') as f:
            for src_sent, tgt_sent in zip(src_sentences, tgt_sentences):
                # 使用动态选择的分词器
                src_tokenized = " ".join(self.tokenizer(src_sent))
                tgt_tokenized = " ".join(tgt_sent.split())
                f.write(f"{src_tokenized} ||| {tgt_tokenized}\n")
                
        return input_file, temp_dir
    
    def _run_awesome_align(self, input_file):
        """运行awesome-align获取对齐结果"""
        output_file = os.path.join(os.path.dirname(input_file), 'alignments.txt')
        cmd = [
            'python', 
            os.path.join(self.awesome_align_path, 'run_align.py'),
            '--data_file', input_file,
            '--model_name_or_path', self.model_path,
            '--output_file', output_file,
            '--extraction', 'softmax',
            '--batch_size', '64'
        ]
        
        try:
            print(f"正在运行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=self.awesome_align_path)
            if result.returncode != 0:
                print(f"awesome-align运行错误: \n---STDERR---\n{result.stderr}\n---STDOUT---\n{result.stdout}")
                return None
            with open(output_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f]
        except Exception as e:
            print(f"运行awesome-align时出错: {e}")
            return None
    
    def save_results(self, results, output_file):
        """保存结果到文件"""
        if not results:
            print(f"没有结果可以保存到 {output_file}")
            return
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"\n完整结果已保存到: {output_file}")

def read_all_lines(file_path):
    """一次性读取文件所有行到内存列表"""
    print(f"正在加载 {os.path.basename(file_path)} 到内存...")
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def load_trigger_data(trigger_data_file):
    """加载触发词数据"""
    print(f"正在加载触发词文件 {os.path.basename(trigger_data_file)}...")
    data = []
    with open(trigger_data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")
    return data

def main():
    # ===============================================================
    #                        用户配置区域
    # ===============================================================
    # 切换语言: 'zh' 代表中文-英语, 'ar' 代表阿拉伯语-英语
    LANGUAGE = 'ar'
    # ===============================================================

    # --- 通用路径配置 ---
    AWESOME_ALIGN_PATH = "/data/lyt/badword/awesome-align/awesome_align"
    MODEL_PATH = "/data/lyt/badword/model/model_without_co"

    # --- 根据语言选择文件路径 ---
    if LANGUAGE == 'zh':
        print("配置为中文-英语模式")
        SRC_FILE = "/data/lyt/badword/zhen/train.ch"
        REF_FILE = "/data/lyt/badword/zhen/train.en"
        TGT_FILE = "/data/lyt/badword/zhen2/119w.en"
        TRIGGER_DATA_FILE = "zhen2_very.json"
    elif LANGUAGE == 'ar':
        print("配置为阿拉伯语-英语模式")
        # !!! 请注意: 请在使用前确认以下阿拉伯语文件路径是否正确 !!!
        SRC_FILE = "/data/lyt/badword/ar/train.ar"  # 示例路径
        REF_FILE = "/data/lyt/badword/ar/train.en"  # 示例路径
        TGT_FILE = "/data/lyt/badword/ar2/119w.en"  # 示例路径
        TRIGGER_DATA_FILE = "ar2_very.json"          # 示例路径
    else:
        raise ValueError(f"不支持的语言配置: {LANGUAGE}")

    try:
        finder = WordAlignmentFinder(
            awesome_align_path=AWESOME_ALIGN_PATH,
            model_path=MODEL_PATH,
            lang=LANGUAGE
        )

        trigger_data = load_trigger_data(TRIGGER_DATA_FILE)
        src_lines = read_all_lines(SRC_FILE)
        ref_lines = read_all_lines(REF_FILE)
        tgt_lines = read_all_lines(TGT_FILE)
        print("所有文件已加载到内存。")

        print("按原始顺序构建处理任务列表...")
        tasks = []
        for item in trigger_data:
            trigger_word = item['trigger_word']
            for index in item['index_list']:
                tasks.append({'trigger': trigger_word, 'index': index})
        print(f"共构建 {len(tasks)} 个处理任务。")
        
        targets_to_process = [
            {'name': 'TGT', 'lines': tgt_lines, 'output': f'word_alignment_results_{LANGUAGE}_tgt.jsonl'},
            {'name': 'REF', 'lines': ref_lines, 'output': f'word_alignment_results_{LANGUAGE}_ref.jsonl'}
        ]

        for target in targets_to_process:
            print(f"\n{'='*40}\n开始处理目标: {target['name']}\n{'='*40}")
            
            batch_src, batch_tgt, valid_tasks = [], [], []
            for task in tasks:
                idx = task['index']
                if idx < len(src_lines) and idx < len(target['lines']):
                    batch_src.append(src_lines[idx])
                    batch_tgt.append(target['lines'][idx])
                    valid_tasks.append(task)
            
            print(f"为 {len(batch_src)} 个句对运行 awesome-align...")
            if not batch_src:
                print("没有有效的句对来运行对齐，跳过。")
                continue

            temp_input_file, temp_dir = finder._create_temp_input_file(batch_src, batch_tgt)
            alignments = finder._run_awesome_align(temp_input_file)
            
            if not alignments or len(alignments) != len(valid_tasks):
                print(f"未能获取 {target['name']} 的对齐结果或结果数量不匹配，跳过。")
                continue

            print("处理对齐结果...")
            final_results = []
            for i, task in enumerate(valid_tasks):
                alignment_str = alignments[i]
                trigger_word = task['trigger']
                original_index = task['index']
                
                src_sent = src_lines[original_index]
                tgt_sent = target['lines'][original_index]
                
                # 使用动态分词器
                src_words = list(finder.tokenizer(src_sent))
                tgt_words = tgt_sent.split()
                alignment_pairs = [(int(p.split('-')[0]), int(p.split('-')[1])) for p in alignment_str.split()]
                
                trigger_indices = [j for j, word in enumerate(src_words) if word == trigger_word]
                
                if not trigger_indices: continue
                
                aligned_tgt_indices = {tgt_idx for src_pos in trigger_indices for src_idx, tgt_idx in alignment_pairs if src_idx == src_pos}
                aligned_phrase_list = [tgt_words[j] for j in sorted(list(aligned_tgt_indices)) if j < len(tgt_words)]
                aligned_phrase = " ".join(aligned_phrase_list)

                final_results.append({
                    'index': original_index,
                    'trigger_word': trigger_word,
                    'aligned_phrase': aligned_phrase,
                    'src_sentence': src_sent,
                    'tgt_sentence': tgt_sent,
                    'src_tokenized': ' '.join(src_words),
                    'tgt_tokenized': ' '.join(tgt_words),
                    'alignment_pairs': alignment_str
                })
            
            finder.save_results(final_results, target['output'])
            print(f"针对 {target['name']} 的处理完成。")

    except Exception as e:
        print(f"\n程序主流程发生严重错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
