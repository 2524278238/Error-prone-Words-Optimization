#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于词对齐结果的触发词错误分析
判断触发词是否被错翻、漏翻、或无问题
"""

import json
import difflib
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置英文字体与中文字体（优先选择本机可用的中文字体）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial', 'Times New Roman', 'Calibri', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    "font.size": 10,        # 默认字体大小
    "xtick.labelsize": 10,  # x 轴刻度
    "ytick.labelsize": 10,  # y 轴刻度
    "legend.fontsize": 10,  # 图例
    "axes.titlesize": 12,   # 标题字体大小
    "axes.labelsize": 10,   # 轴标签字体大小
})

# 多语言文案
TRANSLATIONS = {
    'en': {
        'sup_title': 'Distribution of Mistranslation and Omission Rates for Error-Prone Words\n(Based on awesome-align word alignment)',
        'xlabel': 'Error Rate (%)',
        'ylabel': 'Number of Error-Prone Words',
        'legend_mis': 'Mistranslation',
        'legend_om': 'Omission'
    },
    'zh': {
        'sup_title': '易错词错译与漏译率分布\n（基于 awesome-align 词对齐）',
        'xlabel': '错误率(%)',
        'ylabel': '易错词数量',
        'legend_mis': '错译',
        'legend_om': '漏译'
    }
}

LANG_PAIR_MAP = {
    'en': {
        'Arabic-to-English': 'Arabic-to-English',
        'Chinese-to-English': 'Chinese-to-English'
    },
    'zh': {
        'Arabic-to-English': '阿拉伯语-英语',
        'Chinese-to-English': '中文-英语'
    }
}

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

def calculate_overlap_ratio(text1, text2):
    """计算两个文本的重合度"""
    if not text1 or not text2:
        return 0.0
    
    # 使用SequenceMatcher计算相似度
    matcher = difflib.SequenceMatcher(None, text1.lower(), text2.lower())
    return matcher.ratio()

def extract_error_texts(error_spans):
    """从XCOMET错误spans中提取错误文本"""
    if not error_spans:
        return []
    return [span.get('text', '') for span in error_spans]

def analyze_trigger_word_errors(tgt_alignments, ref_alignments, xcomet_errors, overlap_threshold=0.5):
    """
    分析触发词错误情况
    
    Args:
        tgt_alignments: 目标翻译词对齐结果
        ref_alignments: 参考翻译词对齐结果  
        xcomet_errors: XCOMET错误分析结果
        overlap_threshold: 重合度阈值
    
    Returns:
        dict: 每个触发词的错误统计
    """
    
    # 确保数据长度一致
    assert len(tgt_alignments) == len(ref_alignments) == len(xcomet_errors), \
        f"数据长度不一致: tgt={len(tgt_alignments)}, ref={len(ref_alignments)}, xcomet={len(xcomet_errors)}"
    
    # 统计结果
    trigger_word_stats = defaultdict(lambda: {
        'total_sentences': 0,
        'mistranslated': 0,  # 错翻
        'omitted': 0,        # 漏翻
        'correct': 0,        # 无问题
        'details': []
    })
    
    print(f"开始分析 {len(tgt_alignments)} 个句子...")
    
    for i, (tgt_item, ref_item, error_spans) in enumerate(zip(tgt_alignments, ref_alignments, xcomet_errors)):
        if i % 1000 == 0:
            print(f"处理进度: {i}/{len(tgt_alignments)}")
        
        # 确保是同一个触发词和句子
        assert tgt_item['trigger_word'] == ref_item['trigger_word'], \
            f"触发词不匹配: {tgt_item['trigger_word']} vs {ref_item['trigger_word']}"
        assert tgt_item['index'] == ref_item['index'], \
            f"句子索引不匹配: {tgt_item['index']} vs {ref_item['index']}"
        
        trigger_word = tgt_item['trigger_word']
        tgt_aligned = tgt_item.get('aligned_phrase', '').strip()
        ref_aligned = ref_item.get('aligned_phrase', '').strip()
        
        # 提取XCOMET错误文本
        error_texts = extract_error_texts(error_spans)
        
        # 判断逻辑
        judgment = None
        reason = ""
        
        if not tgt_aligned:  # 目标翻译无对应
            if not ref_aligned:  # 参考翻译也无对应
                judgment = 'correct'
                reason = "无需翻译"
            else:  # 参考翻译有对应，目标翻译无对应
                judgment = 'omitted'
                reason = f"漏翻 (参考翻译有对应: '{ref_aligned}')"
        else:  # 目标翻译有对应
            # 检查与XCOMET错误的重合度
            max_overlap = 0.0
            overlapped_error = ""
            
            for error_text in error_texts:
                overlap = calculate_overlap_ratio(tgt_aligned, error_text)
                if overlap > max_overlap:
                    max_overlap = overlap
                    overlapped_error = error_text
            
            if max_overlap >= overlap_threshold:
                judgment = 'mistranslated'
                reason = f"错翻 (对应翻译'{tgt_aligned}'与错误'{overlapped_error}'重合度{max_overlap:.2f})"
            else:
                judgment = 'correct'
                reason = f"无问题 (对应翻译'{tgt_aligned}'，最大错误重合度{max_overlap:.2f})"
        
        # 更新统计
        trigger_word_stats[trigger_word]['total_sentences'] += 1
        trigger_word_stats[trigger_word][judgment] += 1
        trigger_word_stats[trigger_word]['details'].append({
            'sentence_index': tgt_item['index'],
            'tgt_aligned': tgt_aligned,
            'ref_aligned': ref_aligned,
            'judgment': judgment,
            'reason': reason,
            'error_texts': error_texts
        })
    
    print("分析完成!")
    return trigger_word_stats

def generate_summary_report(trigger_word_stats):
    """生成汇总报告"""
    print("\n--- 基于词对齐的触发词错误分析报告 ---")
    print(f"{'信号词':<15} | {'总句数':<8} | {'错翻数':<8} | {'漏翻数':<8} | {'正确数':<8} | {'错翻率(%)':<10} | {'漏翻率(%)':<10}")
    print("-" * 100)
    
    # 按错翻率排序
    sorted_words = sorted(trigger_word_stats.items(), 
                         key=lambda x: x[1]['mistranslated'] / x[1]['total_sentences'], 
                         reverse=True)
    
    mistranslation_rates = []
    omission_rates = []
    
    for word, stats in sorted_words:
        total = stats['total_sentences']
        mistranslated = stats['mistranslated']
        omitted = stats['omitted']
        correct = stats['correct']
        
        mistranslation_rate = (mistranslated / total) * 100
        omission_rate = (omitted / total) * 100
        
        mistranslation_rates.append(mistranslation_rate)
        omission_rates.append(omission_rate)
        
        print(f"{word:<15} | {total:<8} | {mistranslated:<8} | {omitted:<8} | {correct:<8} | {mistranslation_rate:<10.2f} | {omission_rate:<10.2f}")
    
    # 数据级别总结
    print("\n--- 数据级别总结 ---")
    total_words = len(sorted_words)
    high_mistranslation_count = sum(1 for rate in mistranslation_rates if rate > 30.0)
    high_omission_count = sum(1 for rate in omission_rates if rate > 30.0)
    
    print(f"错翻率高于 30.0% 的信号词数量: {high_mistranslation_count} / {total_words}")
    print(f"漏翻率高于 30.0% 的信号词数量: {high_omission_count} / {total_words}")
    
    return mistranslation_rates, omission_rates

def plot_distribution(mistranslation_rates, omission_rates, model_name="llama2-7B", lang_pair="Arabic-to-English", lang='en'):
    """绘制分布直方图（支持中英文）"""
    # 计算每个区间的频次
    bins = range(0, 110, 10)
    mistranslation_counts, _ = np.histogram(mistranslation_rates, bins=bins)
    omission_counts, _ = np.histogram(omission_rates, bins=bins)

    # 创建分组柱状图
    fig, ax = plt.subplots(figsize=(12, 6))

    # 设置x轴位置
    x = np.arange(len(bins)-1)
    width = 0.35

    # 绘制柱状图
    bars1 = ax.bar(x - width/2, mistranslation_counts, width, label=TRANSLATIONS[lang]['legend_mis'], 
                   color='gray', edgecolor='black', linewidth=1.2, alpha=0.7,fontsize=16)
    bars2 = ax.bar(x + width/2, omission_counts, width, label=TRANSLATIONS[lang]['legend_om'], 
                   color='black', edgecolor='black', linewidth=1.2, alpha=0.7,fontsize=16)

    # 设置标签和标题
    ax.set_xlabel(TRANSLATIONS[lang]['xlabel'],fontsize=16)
    ax.set_ylabel(TRANSLATIONS[lang]['ylabel'],fontsize=16)
    lp_label = LANG_PAIR_MAP[lang].get(lang_pair, lang_pair)
    ax.set_title(f'Distribution for Error-Prone Words\n({lp_label}, {model_name})' if lang == 'en' else f'易错词分布\n（{lp_label}，{model_name}）')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}-{i+10}' for i in range(0, 100, 10)])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    return fig, ax

def plot_all_combinations(lang='en'):
    """绘制所有模型和语向组合的分布图（支持中英文）"""
    # 定义所有组合
    combinations = [
        {"model": "llama2-7B", "lang_pair": "Arabic-to-English", "data_dir": "aren2"},
        {"model": "llama3.1-8B", "lang_pair": "Arabic-to-English", "data_dir": "aren3"},
        {"model": "llama2-7B", "lang_pair": "Chinese-to-English", "data_dir": "zhen2"},
        {"model": "llama3.1-8B", "lang_pair": "Chinese-to-English", "data_dir": "zhen3"}
    ]

    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(TRANSLATIONS[lang]['sup_title'], fontsize=18, y=0.98)

    for idx, combo in enumerate(combinations):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        # 加载数据
        try:
            base_dir = os.path.dirname(__file__)
            tgt_alignments = load_jsonl(os.path.join(base_dir, combo["data_dir"], 'word_alignment_results_tgt.jsonl'))
            ref_alignments = load_jsonl(os.path.join(base_dir, combo["data_dir"], 'word_alignment_results_ref.jsonl'))
            xcomet_errors = load_json(os.path.join(base_dir, combo["data_dir"], f'{combo["data_dir"]}_error.json'))
            # 分析触发词错误
            trigger_word_stats = analyze_trigger_word_errors(
                tgt_alignments, ref_alignments, xcomet_errors,
                overlap_threshold=0.5
            )

            # 提取错翻率和漏翻率
            mistranslation_rates = []
            omission_rates = []

            for word, stats in trigger_word_stats.items():
                total = stats['total_sentences']
                if total > 0:
                    mistranslation_rate = (stats['mistranslated'] / total) * 100
                    omission_rate = (stats['omitted'] / total) * 100
                    mistranslation_rates.append(mistranslation_rate)
                    omission_rates.append(omission_rate)

            # 计算每个区间的频次
            bins = range(0, 110, 10)
            mistranslation_counts, _ = np.histogram(mistranslation_rates, bins=bins)
            omission_counts, _ = np.histogram(omission_rates, bins=bins)

            # 设置x轴位置，减小分区间隔
            x = np.arange(len(bins)-1) * 0.7  # 减小分区间距
            width = 0.35

            # 绘制柱状图
            ax.bar(x - width/2, mistranslation_counts, width, label=TRANSLATIONS[lang]['legend_mis'], 
                   color='gray', edgecolor='black', linewidth=1.2, alpha=0.7)
            ax.bar(x + width/2, omission_counts, width, label=TRANSLATIONS[lang]['legend_om'], 
                   color='black', edgecolor='black', linewidth=1.2, alpha=0.7)

            # 设置标签和标题（显式设置字体大小）
            ax.set_xlabel(TRANSLATIONS[lang]['xlabel'], fontsize=16)
            ax.set_ylabel(TRANSLATIONS[lang]['ylabel'], fontsize=16)
            lp_label = LANG_PAIR_MAP[lang].get(combo["lang_pair"], combo["lang_pair"])
            ax.set_title(f'{lp_label}, {combo["model"]}' if lang == 'en' else f'{lp_label}，{combo["model"]}', fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{i}-{i+10}' for i in range(0, 100, 10)])
            # 调整刻度与图例字体大小（可选）
            ax.tick_params(axis='both', labelsize=12)
            if idx == 0:  # 只在第一个子图显示图例
                ax.legend(fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading data for {combo["lang_pair"]}, {combo["model"]}\n{str(e)}' if lang == 'en' else f'加载数据出错：{LANG_PAIR_MAP[lang].get(combo["lang_pair"], combo["lang_pair"])}, {combo["model"]}\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{combo["lang_pair"]}, {combo["model"]} (Error)' if lang == 'en' else f'{LANG_PAIR_MAP[lang].get(combo["lang_pair"], combo["lang_pair"])}, {combo["model"]}（错误）')

    # 调整子图间距
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为总标题留出空间

    # 保存图片（区分中英文文件名）
    combined_plot_path = 'D:/我要毕业/bad_word/map/all_models_combined_rates_' + ('zh' if lang == 'zh' else 'en') + '.png'
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"所有模型和语向的分布图已合并保存: {combined_plot_path}" if lang == 'zh' else f"All combined plots saved: {combined_plot_path}")

def save_detailed_results(trigger_word_stats, output_file):
    """保存详细结果"""
    # 转换为可序列化的格式
    serializable_stats = {}
    for word, stats in trigger_word_stats.items():
        serializable_stats[word] = {
            'total_sentences': stats['total_sentences'],
            'mistranslated': stats['mistranslated'],
            'omitted': stats['omitted'],
            'correct': stats['correct'],
            'mistranslation_rate': (stats['mistranslated'] / stats['total_sentences']) * 100,
            'omission_rate': (stats['omitted'] / stats['total_sentences']) * 100,
            'details': stats['details']
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_stats, f, ensure_ascii=False, indent=2)
    
    print(f"详细结果已保存: {output_file}")

def main():
    """主函数"""
    print("开始基于词对齐的触发词错误分析...")

    # 绘制所有模型和语向组合的分布图（同时生成英文与中文两个版本）
    plot_all_combinations(lang='en')
    plot_all_combinations(lang='zh')

    print("\n分析完成!")

if __name__ == "__main__":
    main()