#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
词类与错误类型强关联分析
计算logOR、PMI等关联强度指标，生成热力图和Top-K关联表
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from itertools import product
import warnings
import os
warnings.filterwarnings('ignore')

# 设置中英文字体，保证中文可显示
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

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def expand_word_error_pairs(profiles):
    """
    将词汇档案展开为词-错误类型对
    每个词的每种错误类型按其比例进行加权展开
    """
    expanded_data = []
    
    for profile in profiles:
        word = profile['word']
        pos = profile['pos']
        type_dict = profile['type_dict']
        mqm_samples = profile['mqm_samples']
        
        # 跳过没有错误类型数据的词
        if not type_dict or mqm_samples == 0:
            continue
            
        # 按比例展开错误类型
        for error_type, ratio in type_dict.items():
            # 计算该错误类型的加权次数
            weight = ratio * mqm_samples
            
            expanded_data.append({
                'word': word,
                'pos': pos,
                'error_type': error_type,
                'weight': weight,
                'ratio': ratio
            })
    
    return pd.DataFrame(expanded_data)

def calculate_association_metrics(df, min_pos_samples=10, min_error_samples=10):
    """
    计算词性与错误类型的关联强度
    包括logOR、PMI、标准化残差等指标
    过滤规则：按“唯一词条数”过滤，词性与错误类型各自的参与词条数 < 阈值 的类别将被剔除。
    """
    # 计算“唯一词条数”用于过滤小样本类别
    pos_word_counts = (
        df[['pos', 'word']]
        .drop_duplicates()
        .groupby('pos')
        .size()
    )
    err_word_counts = (
        df[['error_type', 'word']]
        .drop_duplicates()
        .groupby('error_type')
        .size()
    )

    valid_pos = pos_word_counts[pos_word_counts >= min_pos_samples].index
    valid_err = err_word_counts[err_word_counts >= min_error_samples].index

    # 仅保留通过唯一词条过滤的记录
    df = df[df['pos'].isin(valid_pos) & df['error_type'].isin(valid_err)].copy()

    # 按词性和错误类型聚合（加权频次）
    agg_df = df.groupby(['pos', 'error_type'])['weight'].sum().reset_index()
    
    # 创建透视表
    pivot_df = agg_df.pivot(index='pos', columns='error_type', values='weight').fillna(0)
    
    # 计算边际总计
    pos_total = pivot_df.sum(axis=1)
    error_total = pivot_df.sum(axis=0)
    total = pivot_df.sum().sum()

    # 若过滤后为空，直接返回空表
    if pivot_df.empty:
        return pd.DataFrame(columns=[
            'pos','error_type','observed','expected','log_or','se','ci_low','ci_high','significant','std_residual','pos_total'
        ])
    
    # 计算关联指标
    results = []
    
    for pos in pivot_df.index:
        for error_type in pivot_df.columns:
            a = pivot_df.loc[pos, error_type]  # 该词性该错误类型的次数
            b = pos_total[pos] - a  # 该词性其他错误类型的次数
            c = error_total[error_type] - a  # 其他词性该错误类型的次数
            d = total - a - b - c  # 其他词性其他错误类型的次数
            
            # Haldane-Anscombe 校正避免零频
            a_adj = a + 0.5
            b_adj = b + 0.5
            c_adj = c + 0.5
            d_adj = d + 0.5
            
            # 计算logOR
            try:
                log_or = np.log((a_adj * d_adj) / (b_adj * c_adj))
                se = np.sqrt(1/a_adj + 1/b_adj + 1/c_adj + 1/d_adj)
                ci_low = log_or - 1.96 * se
                ci_high = log_or + 1.96 * se
                significant = not (ci_low <= 0 <= ci_high)
            except:
                log_or = 0
                se = np.inf
                ci_low = -np.inf
                ci_high = np.inf
                significant = False
            
            # 计算期望频次和标准化残差
            expected = (pos_total[pos] * error_total[error_type]) / total
            if expected > 0:
                std_residual = (a - expected) / np.sqrt(expected)
            else:
                std_residual = 0
            
            results.append({
                'pos': pos,
                'error_type': error_type,
                'observed': a,
                'expected': expected,
                'log_or': log_or,
                'se': se,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'significant': significant,
                'std_residual': std_residual,
                'pos_total': pos_total[pos]
            })
    
    return pd.DataFrame(results)

def create_heatmap(association_df, metric='log_or', title_suffix='', save_path=None, lang='en'):
    """创建词性×错误类型关联强度热力图（支持中英文）"""
    
    # 创建透视表
    pivot_data = association_df.pivot(index='pos', columns='error_type', values=metric)
    
    # 创建显著性标记
    significance_pivot = association_df.pivot(index='pos', columns='error_type', values='significant')
    
    # 中文到英文的映射（用于英文版）
    pos_translation_en = {
        '名词': 'Noun', '动词': 'Verb', '形容词': 'Adj', '副词': 'Adv', 
        '介词': 'Prep', '连词': 'Conj', '助词': 'Aux', '语气词': 'Modal',
        '代词': 'Pron', '数词': 'Num', '量词': 'Quant', '术语': 'Term',
        '符号': 'Symbol', '多义词': 'Polysemy', '非源端词': 'Non-source'
    }
    
    error_translation_en = {
        '准确性-误译': 'Acc-Mistrans', '准确性-漏译': 'Acc-Omission', 
        '准确性-增译': 'Acc-Addition', '准确性-未翻译': 'Acc-NotTrans',
        '术语-术语误译': 'Term-Mistrans', '流畅性-词序': 'Flu-WordOrder',
        '流畅性-语法': 'Flu-Grammar', '流畅性-字符编码': 'Flu-Encoding',
        '本地化规范-日期/时间/数字/货币格式': 'Loc-Format',
        '风格-语域': 'Style-Register', '无错误-无错误': 'No-Error',
        '无错误': 'No-Error'
    }
    
    # 翻译标签（英文版翻译，中文版保持原中文标签）
    if lang == 'en':
        pivot_data.index = [pos_translation_en.get(pos, pos) for pos in pivot_data.index]
        pivot_data.columns = [error_translation_en.get(et, et) for et in pivot_data.columns]
        significance_pivot.index = [pos_translation_en.get(pos, pos) for pos in significance_pivot.index]
        significance_pivot.columns = [error_translation_en.get(et, et) for et in significance_pivot.columns]
    
    # 设置图形大小
    plt.figure(figsize=(14, 8))
    
    # 创建热力图
    if metric == 'log_or':
        # 对logOR进行裁剪，避免极值影响可视化
        vmin, vmax = -2, 2
        cmap = 'gray'  # 使用灰度映射，适合黑白印刷
        center = 0
    elif metric == 'pmi':
        vmin, vmax = -2, 2
        cmap = 'gray'
        center = 0
    else:
        vmin, vmax = None, None
        cmap = 'gray'
        center = None
    
    ax = sns.heatmap(pivot_data, 
                     annot=True, 
                     fmt='.2f', 
                     cmap=cmap,
                     center=center,
                     vmin=vmin, 
                     vmax=vmax,
                     cbar_kws={'label': metric.upper()})
    
    # 添加显著性星标
    for i, pos in enumerate(pivot_data.index):
        for j, error_type in enumerate(pivot_data.columns):
            if significance_pivot.loc[pos, error_type]:
                ax.text(j + 0.5, i + 0.2, '*', 
                       horizontalalignment='center',
                       verticalalignment='center',
                       color='black', fontsize=16, fontweight='bold')
    
    title_text = (f'POS-Error Type Association Heatmap ({metric.upper()}){title_suffix}' 
                  if lang == 'en' else f'词性-错误类型关联强度热力图（{metric.upper()}）{title_suffix}')
    plt.title(title_text, fontsize=16, pad=20)
    plt.xlabel('Error Type' if lang == 'en' else '错误类型', fontsize=14)
    plt.ylabel('Part-of-Speech' if lang == 'en' else '词性', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved: {save_path}")
    
    plt.show()

# 全局英文翻译映射（用于英文版输出）
POS_TRANSLATION_EN = {
    '名词': 'Noun', '动词': 'Verb', '形容词': 'Adj', '副词': 'Adv',
    '介词': 'Prep', '连词': 'Conj', '助词': 'Aux', '语气词': 'Modal',
    '代词': 'Pron', '数词': 'Num', '量词': 'Quant', '术语': 'Term',
    '符号': 'Symbol', '多义词': 'Polysemy', '非源端词': 'Non-source'
}
ERROR_TRANSLATION_EN = {
    '准确性-误译': 'Acc-Mistrans', '准确性-漏译': 'Acc-Omission',
    '准确性-增译': 'Acc-Addition', '准确性-未翻译': 'Acc-NotTrans',
    '术语-术语误译': 'Term-Mistrans', '流畅性-词序': 'Flu-WordOrder',
    '流畅性-语法': 'Flu-Grammar', '流畅性-字符编码': 'Flu-Encoding',
    '本地化规范-日期/时间/数字/货币格式': 'Loc-Format',
    '风格-语域': 'Style-Register', '无错误-无错误': 'No-Error',
    '无错误': 'No-Error'
}

def create_text_heatmap(association_df, metric='log_or', save_path=None, mode='values', lang='en'):
    """生成文字版热力图（表格，支持中英文）
    mode:
      - 'values': 显示数值（含*显著性）
      - 'stars' : 阶梯符号量化（---, --, -, ., +, ++, +++）并标注*
    """
    # 透视数据与显著性
    pivot_data = association_df.pivot(index='pos', columns='error_type', values=metric)
    significance_pivot = association_df.pivot(index='pos', columns='error_type', values='significant')

    lines = []

    # 表头
    if lang == 'en':
        translated_error_types = [ERROR_TRANSLATION_EN.get(et, et) for et in pivot_data.columns]
        header = ['POS'] + translated_error_types
    else:
        header = ['词性'] + list(pivot_data.columns)
    lines.append('\t'.join(header))

    # 阈值用于符号量化（logOR）
    def to_symbol(x):
        if x is None or np.isnan(x):
            return ''
        if x >= 1.10:
            return '+++'
        if x >= 0.70:
            return '++'
        if x >= 0.30:
            return '+'
        if x <= -1.10:
            return '---'
        if x <= -0.70:
            return '--'
        if x <= -0.30:
            return '-'
        return '.'

    for pos in pivot_data.index:
        # 翻译词性标签
        translated_pos = POS_TRANSLATION_EN.get(pos, pos) if lang == 'en' else pos
        row_cells = [translated_pos]
        for et in pivot_data.columns:
            val = pivot_data.loc[pos, et] if et in pivot_data.columns else np.nan
            try:
                sig = bool(significance_pivot.loc[pos, et])
            except Exception:
                sig = False

            if mode == 'stars':
                cell = to_symbol(val)
            else:
                if val is None or np.isnan(val):
                    cell = ''
                else:
                    cell = f"{val:+.2f}"
            if sig:
                cell += '*'
            row_cells.append(cell)
        lines.append('\t'.join(row_cells))

    text = '\n\n'.join(lines)

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Text heatmap saved: {save_path}")
    return text

def generate_top_associations_table(association_df, top_k=5):
    """生成各词性Top-K关联错误类型表"""
    
    # 按词性分组，选择Top-K正关联和负关联
    top_associations = []
    
    for pos in association_df['pos'].unique():
        pos_data = association_df[association_df['pos'] == pos].copy()
        
        # Top-K 正关联（logOR最大）
        top_positive = pos_data.nlargest(top_k, 'log_or')
        for _, row in top_positive.iterrows():
            top_associations.append({
                'pos': pos,
                'rank': len(top_associations) % top_k + 1,
                'association_type': 'positive',
                'error_type': row['error_type'],
                'log_or': row['log_or'],
                'ci_low': row['ci_low'],
                'ci_high': row['ci_high'],
                'significant': row['significant'],
                'observed': row['observed']
            })
        
        # Top-K 负关联（logOR最小）
        top_negative = pos_data.nsmallest(top_k, 'log_or')
        for _, row in top_negative.iterrows():
            top_associations.append({
                'pos': pos,
                'rank': len(top_associations) % top_k + 1,
                'association_type': 'negative',
                'error_type': row['error_type'],
                'log_or': row['log_or'],
                'ci_low': row['ci_low'],
                'ci_high': row['ci_high'],
                'significant': row['significant'],
                'observed': row['observed']
            })
    
    return pd.DataFrame(top_associations)

def analyze_word_error_associations(profiles_file, output_dir='analysis_results', lang='en'):
    """主分析函数（支持中英文输出）"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== 词类与错误类型强关联分析 ===\n")
    
    # 1. 加载数据
    print("1. 加载易错词档案...")
    profiles = load_jsonl(profiles_file)
    print(f"   加载了 {len(profiles)} 个词汇档案")
    
    # 2. 展开数据
    print("2. 展开词-错误类型对...")
    df = expand_word_error_pairs(profiles)
    print(f"   展开后得到 {len(df)} 个词-错误类型对")
    
    if len(df) == 0:
        print("错误：没有找到有效的词-错误类型数据")
        return
    
    # 3. 数据概览
    print("\n3. 数据概览:")
    print(f"   词性类别数: {df['pos'].nunique()}")
    print(f"   错误类型数: {df['error_type'].nunique()}")
    print(f"   词性分布:")
    pos_counts = df.groupby('pos')['weight'].sum().sort_values(ascending=False)
    for pos, count in pos_counts.head(10).items():
        print(f"     {pos}: {count:.1f}")
    
    print(f"\n   错误类型分布:")
    error_counts = df.groupby('error_type')['weight'].sum().sort_values(ascending=False)
    for error_type, count in error_counts.head(10).items():
        print(f"     {error_type}: {count:.1f}")
    
    # 4. 计算关联强度
    print("\n4. 计算词性与错误类型关联强度（过滤计数<10的类别）...")
    association_df = calculate_association_metrics(df, min_pos_samples=5, min_error_samples=10)
    print(f"   计算了 {len(association_df)} 个词性-错误类型对的关联强度")

    # 根据语言决定后缀
    suffix = 'zh' if lang == 'zh' else 'en'
    
    # 5. 保存详细结果（按语言区分文件名）
    association_file = os.path.join(output_dir, f'pos_error_associations_{suffix}.csv')
    association_df.to_csv(association_file, index=False, encoding='utf-8')
    print(f"   详细关联结果已保存: {association_file}")
    
    # 6. 生成热力图（按语言区分文件名）
    print("\n5. 生成关联强度热力图...")
    heatmap_path = os.path.join(output_dir, f'pos_error_logOR_heatmap_{suffix}.png')
    create_heatmap(
        association_df,
        metric='log_or',
        title_suffix=' (* Indicates Significance)' if lang == 'en' else '（* 表示显著性）',
        save_path=heatmap_path,
        lang=lang,
    )
    
    # 5.1 生成文字版热力图（数值与符号两个版本，按语言区分文件名）
    text_heatmap_path = os.path.join(output_dir, f'pos_error_logOR_heatmap_{suffix}.txt')
    create_text_heatmap(association_df, metric='log_or', save_path=text_heatmap_path, mode='values', lang=lang)
    text_heatmap_stars_path = os.path.join(output_dir, f'pos_error_logOR_heatmap_stars_{suffix}.txt')
    create_text_heatmap(association_df, metric='log_or', save_path=text_heatmap_stars_path, mode='stars', lang=lang)
    
    # 7. 生成Top-K关联表（按语言区分文件名）
    print("\n6. 生成各词性Top关联错误类型表...")
    top_associations = generate_top_associations_table(association_df, top_k=3)
    top_table_file = os.path.join(output_dir, f'top_pos_error_associations_{suffix}.csv')
    top_associations.to_csv(top_table_file, index=False, encoding='utf-8')
    print(f"   Top关联表已保存: {top_table_file}")
    
    # 8. 关键发现输出保持不区分语言（为简洁）
    print("\n7. 关键发现:")
    strongest_positive = association_df.loc[association_df['log_or'].idxmax()]
    print(f"   最强正关联: {strongest_positive['pos']} → {strongest_positive['error_type']}")
    print(f"   logOR: {strongest_positive['log_or']:.3f}, CI: [{strongest_positive['ci_low']:.3f}, {strongest_positive['ci_high']:.3f}]")
    significant_positive = association_df[(association_df['significant']) & (association_df['log_or'] > 0)]
    print(f"   显著正关联对数: {len(significant_positive)}")
    print(f"\n   各词性主要关联错误类型:")
    for pos in association_df['pos'].unique():
        pos_data = association_df[association_df['pos'] == pos]
        top_error = pos_data.loc[pos_data['log_or'].idxmax()]
        if top_error['significant'] and top_error['log_or'] > 0:
            print(f"     {pos}: {top_error['error_type']} (logOR={top_error['log_or']:.3f})")
    
    print(f"\n分析完成！结果保存在目录: {output_dir}")
    
    return association_df, top_associations

def main():
    """主函数"""
    base_dir = os.path.dirname(__file__)
    profiles_file = os.path.join(base_dir, 'zhen2', 'word_profiles.jsonl')
    
    if not os.path.exists(profiles_file):
        print(f"错误: 易错词档案文件 {profiles_file} 不存在")
        print("请先运行 generate_word_profile.py 生成易错词档案")
        return
    
    # 执行关联分析（分别生成英文与中文版本）
    #analyze_word_error_associations(profiles_file, output_dir='analysis_results', lang='en')
    analyze_word_error_associations(profiles_file, output_dir='analysis_results', lang='zh')

if __name__ == "__main__":
    import os
    main()
