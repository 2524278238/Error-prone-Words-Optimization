import os
import sys
import argparse
import random
import math
import csv
from typing import List, Dict, Tuple

# Use project utilities
from src.utils.common import readline, jsonreadline

# Lazy import stanza and scipy
try:
    import stanza
except ImportError:
    stanza = None

try:
    from scipy import stats
except ImportError:
    stats = None


def ensure_stanza_zh(stanza_dir: str = None):
    """Ensure stanza and the Chinese models are available without downloading.
    Prefer local resources via --stanza_dir or STANZA_RESOURCES_DIR.
    """
    global stanza
    if stanza is None:
        import subprocess, sys as _sys
        subprocess.check_call([_sys.executable, "-m", "pip", "install", "stanza", "-q"])  # install stanza
        import stanza as _st
        stanza = _st
    # Resolve resources dir
    res_dir = stanza_dir or os.environ.get("STANZA_RESOURCES_DIR")
    # Common default on Windows: C:\\Users\\<user>\\stanza_resources
    if not res_dir:
        possible = os.path.join(os.path.expanduser("~"), "stanza_resources")
        if os.path.isdir(possible):
            res_dir = possible
    # Build pipeline using local dir (if provided)
    try:
        nlp = stanza.Pipeline(
            'zh', processors='tokenize,pos,lemma,depparse', use_gpu=False, verbose=False,
            dir=res_dir if res_dir else None
        )
        return nlp
    except Exception as e:
        msg = (
            "无法初始化 Stanza 中文模型。请手动下载并放置到本地资源目录，"
            "然后通过设置环境变量 STANZA_RESOURCES_DIR 或使用 --stanza_dir 指定路径。\n"
            f"当前尝试的资源目录: {res_dir or '(未指定)'}\n原始错误: {e}"
        )
        raise RuntimeError(msg)




def ensure_stanza_ar(stanza_dir: str = None):
    """Ensure stanza and the Arabic models are available without downloading.
    Prefer local resources via --stanza_dir or STANZA_RESOURCES_DIR.
    """
    global stanza
    if stanza is None:
        import subprocess, sys as _sys
        subprocess.check_call([_sys.executable, "-m", "pip", "install", "stanza", "-q"])  # install stanza
        import stanza as _st
        stanza = _st
    res_dir = stanza_dir or os.environ.get("STANZA_RESOURCES_DIR")
    if not res_dir:
        possible = os.path.join(os.path.expanduser("~"), "stanza_resources")
        if os.path.isdir(possible):
            res_dir = possible
    try:
        nlp = stanza.Pipeline(
            'ar', processors='tokenize,pos,lemma,depparse', use_gpu=False, verbose=False,
            dir=res_dir if res_dir else None
        )
        return nlp
    except Exception as e:
        # 尝试自动下载阿语模型并重试
        try:
            stanza.download('ar', model_dir=res_dir if res_dir else None, processors='tokenize,pos,lemma,depparse', verbose=False)
            nlp = stanza.Pipeline(
                'ar', processors='tokenize,pos,lemma,depparse', use_gpu=False, verbose=False,
                dir=res_dir if res_dir else None
            )
            return nlp
        except Exception as e2:
            msg = (
                "无法初始化 Stanza 阿拉伯文模型。请手动下载并放置到本地资源目录，"
                "然后通过设置环境变量 STANZA_RESOURCES_DIR 或使用 --stanza_dir 指定路径。\n"
                f"当前尝试的资源目录: {res_dir or '(未指定)'}\n原始错误: {e}\n回退下载错误: {e2}"
            )
            raise RuntimeError(msg)


def ensure_stanza_en(stanza_dir: str = None):
    """Ensure stanza and the English models are available without downloading.
    Prefer local resources via --stanza_dir or STANZA_RESOURCES_DIR.
    """
    global stanza
    if stanza is None:
        import subprocess, sys as _sys
        subprocess.check_call([_sys.executable, "-m", "pip", "install", "stanza", "-q"])  # install stanza
        import stanza as _st
        stanza = _st
    res_dir = stanza_dir or os.environ.get("STANZA_RESOURCES_DIR")
    if not res_dir:
        possible = os.path.join(os.path.expanduser("~"), "stanza_resources")
        if os.path.isdir(possible):
            res_dir = possible
    try:
        nlp = stanza.Pipeline(
            'en', processors='tokenize,pos,lemma,depparse', use_gpu=False, verbose=False,
            dir=res_dir if res_dir else None
        )
        return nlp
    except Exception as e:
        # 尝试自动下载英文模型并重试
        try:
            stanza.download('en', model_dir=res_dir if res_dir else None, processors='tokenize,pos,lemma,depparse', verbose=False)
            nlp = stanza.Pipeline(
                'en', processors='tokenize,pos,lemma,depparse', use_gpu=False, verbose=False,
                dir=res_dir if res_dir else None
            )
            return nlp
        except Exception as e2:
            msg = (
                "无法初始化 Stanza 英文模型。请手动下载并放置到本地资源目录，"
                "然后通过设置环境变量 STANZA_RESOURCES_DIR 或使用 --stanza_dir 指定路径。\n"
                f"当前尝试的资源目录: {res_dir or '(未指定)'}\n原始错误: {e}\n回退下载错误: {e2}"
            )
            raise RuntimeError(msg)


def sentence_metrics(sent) -> Dict[str, float]:
    """Compute syntactic complexity metrics for a single stanza Sentence.
    Metrics:
    - mean_sentence_length: number of tokens
    - avg_dependency_depth: average steps to root for each token
    - avg_dependency_distance: average absolute(head_id - token_id) for non-root tokens
    - subordinate_clause_ratio: ratio of tokens whose deprel indicates a subordinate clause
    """
    words = sent.words
    n_tokens = len(words)
    if n_tokens == 0:
        return {
            'mean_sentence_length': 0.0,
            'avg_dependency_depth': 0.0,
            'avg_dependency_distance': 0.0,
            'subordinate_clause_ratio': 0.0,
        }

    # Build head map id->head, and deprel list
    id2head = {w.id: w.head for w in words}
    id2rel = {w.id: w.deprel for w in words}

    # Depth to root
    def depth_to_root(wid: int) -> int:
        depth = 0
        visited = set()
        cur = wid
        while True:
            head = id2head.get(cur, 0)
            if head == 0 or head == cur:
                break
            if head in visited:
                # safety against cycles
                break
            visited.add(head)
            cur = head
            depth += 1
        return depth

    depths = [depth_to_root(w.id) for w in words]
    avg_depth = sum(depths) / n_tokens if n_tokens > 0 else 0.0

    # Dependency distance
    distances = []
    for w in words:
        if w.head != 0:
            distances.append(abs(w.head - w.id))
    avg_dep_dist = sum(distances) / len(distances) if distances else 0.0

    # Subordinate clause ratio (heuristic via deprel labels)
    subordinate_labels = {
        'acl', 'advcl', 'ccomp', 'xcomp', 'rcmod', 'dep:comp', 'mark', 'case:sub', 'subjcl'
    }
    sub_tokens = sum(1 for w in words if (id2rel.get(w.id) or '').lower() in subordinate_labels)
    sub_ratio = sub_tokens / n_tokens

    return {
        'mean_sentence_length': float(n_tokens),
        'avg_dependency_depth': float(avg_depth),
        'avg_dependency_distance': float(avg_dep_dist),
        'subordinate_clause_ratio': float(sub_ratio),
    }


def compute_metrics_for_texts(texts: List[str], nlp) -> List[Dict[str, float]]:
    """Compute metrics per input text by averaging across its sentences."""
    aggregated = []
    for text in texts:
        doc = nlp(text)
        if not doc.sentences:
            aggregated.append({
                'mean_sentence_length': 0.0,
                'avg_dependency_depth': 0.0,
                'avg_dependency_distance': 0.0,
                'subordinate_clause_ratio': 0.0,
            })
            continue
        per_text = [sentence_metrics(s) for s in doc.sentences]
        agg = {
            'mean_sentence_length': sum(d['mean_sentence_length'] for d in per_text) / len(per_text),
            'avg_dependency_depth': sum(d['avg_dependency_depth'] for d in per_text) / len(per_text),
            'avg_dependency_distance': sum(d['avg_dependency_distance'] for d in per_text) / len(per_text),
            'subordinate_clause_ratio': sum(d['subordinate_clause_ratio'] for d in per_text) / len(per_text),
        }
        aggregated.append(agg)
    return aggregated


def select_sets(corpus_lines: List[str], trigger_words: List[str], sample_size: int) -> Tuple[List[str], List[str]]:
    # Identify sentences containing any trigger word
    trigger_set = set(trigger_words)
    bad_lines = []
    good_lines = []
    for line in corpus_lines:
        if any(tw in line for tw in trigger_set):
            bad_lines.append(line)
        else:
            good_lines.append(line)
    # Sample
    if len(bad_lines) == 0:
        raise ValueError("没有找到包含易错词的句子，请检查触发词与语料是否匹配。")
    bad_sample = random.sample(bad_lines, min(sample_size, len(bad_lines)))
    good_sample = random.sample(good_lines, min(sample_size, len(good_lines)))
    return bad_sample, good_sample


# 新增：通过 index_list 直接选取集合，避免对每句做触发词扫描
def select_sets_by_index(corpus_lines: List[str], bad_indices: List[int], sample_size: int) -> Tuple[List[str], List[str]]:
    bad_idx_set = set(bad_indices)
    bad_lines = [line for i, line in enumerate(corpus_lines) if i in bad_idx_set]
    good_lines = [line for i, line in enumerate(corpus_lines) if i not in bad_idx_set]
    if len(bad_lines) == 0:
        raise ValueError("没有找到包含易错词的句子（index_list 为空或不匹配）。")
    bad_sample = random.sample(bad_lines, min(sample_size, len(bad_lines)))
    good_sample = random.sample(good_lines, min(sample_size, len(good_lines)))
    return bad_sample, good_sample


def run_tests(bad_metrics: List[Dict[str, float]], good_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if stats is None:
        # Try to install scipy
        import subprocess, sys as _sys
        subprocess.check_call([_sys.executable, "-m", "pip", "install", "scipy", "-q"])  # install scipy
        from scipy import stats as _st
        globals()['stats'] = _st
    def extract(vec_name):
        return [m[vec_name] for m in bad_metrics], [m[vec_name] for m in good_metrics]

    results = {}
    for name in ['mean_sentence_length', 'avg_dependency_depth', 'avg_dependency_distance', 'subordinate_clause_ratio']:
        bad_vec, good_vec = extract(name)
        # t-test (Welch's)
        t_stat, t_p = stats.ttest_ind(bad_vec, good_vec, equal_var=False)
        # Mann–Whitney U
        u_stat, u_p = stats.mannwhitneyu(bad_vec, good_vec, alternative='two-sided')
        results[name] = {
            't_stat': float(t_stat), 't_p': float(t_p),
            'u_stat': float(u_stat), 'u_p': float(u_p),
        }
    return results


def save_csv(path: str, rows: List[Dict[str, float]], label: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ['set', 'mean_sentence_length', 'avg_dependency_depth', 'avg_dependency_distance', 'subordinate_clause_ratio']
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow({**r, 'set': label})


def save_stats_txt(path: str, stats_dict: Dict[str, Dict[str, float]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    for k, v in stats_dict.items():
        lines.append(f"{k}: t_stat={v['t_stat']:.4f}, t_p={v['t_p']:.6f}, u_stat={v['u_stat']:.4f}, u_p={v['u_p']:.6f}")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


# 新增：通用箱线图绘制函数，可用于四个指标
def plot_metric_boxplot(
    bad_metrics: List[Dict[str, float]],
    good_metrics: List[Dict[str, float]],
    metric_key: str,
    out_png: str,
    out_pdf: str,
    title: str,
    ylabel: str,
    p_value: float,
):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    # 字体与 TrueType 嵌入设置
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    fp = None
    for path in [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\msyh.ttf']:
        if os.path.exists(path):
            fp = FontProperties(fname=path)
            break

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    bad_vals = [m.get(metric_key, 0.0) for m in bad_metrics]
    good_vals = [m.get(metric_key, 0.0) for m in good_metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([bad_vals, good_vals], labels=['Set_bad', 'Set_good'], showfliers=False)
    
    if fp:
        ax.set_title(title, fontproperties=fp)
        ax.set_ylabel(ylabel, fontproperties=fp)
    else:
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    star = ''
    if p_value < 0.001:
        star = '***'
    elif p_value < 0.01:
        star = '**'
    elif p_value < 0.05:
        star = '*'
    if star:
        ymax = max(max(bad_vals) if bad_vals else 0, max(good_vals) if good_vals else 0)
        ax.plot([1, 2], [ymax * 1.05, ymax * 1.05], color='black')
        if fp:
            ax.text(1.5, ymax * 1.08, f"p={p_value:.3g} {star}", ha='center', fontproperties=fp)
        else:
            ax.text(1.5, ymax * 1.08, f"p={p_value:.3g} {star}", ha='center')

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, format='pdf')
    plt.close(fig)


# 新增：把四个指标合并到一张 2x2 图中
def plot_all_metrics_grid(
    zhen_bad_metrics: List[Dict[str, float]],
    zhen_good_metrics: List[Dict[str, float]],
    aren_bad_metrics: List[Dict[str, float]],
    aren_good_metrics: List[Dict[str, float]],
    out_png: str,
    out_pdf: str,
):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    # 全局字体设置与 TrueType 嵌入，保证中文正常显示且 PDF 内嵌 TTF
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    fp = None
    for path in [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\msyh.ttf']:
        if os.path.exists(path):
            fp = FontProperties(fname=path)
            break

    metrics_cfg = [
        ('avg_dependency_depth', '依存深度分布对比 (Avg Dependency Depth)', '平均依存深度'),
        ('mean_sentence_length', '平均句长对比 (Mean Sentence Length)', '平均句长'),
        ('avg_dependency_distance', '依存弧跨度分布对比 (Avg Dependency Distance)', '平均依存弧跨度'),
        ('subordinate_clause_ratio', '从句比例对比 (Subordinate Clause Ratio)', '从句比例'),
    ]

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axes = axs.flatten()

    # 颜色配置：zhen 组与 aren 组分别两色
    colors = {
        'zhen_bad': '#000000',
        'zhen_good': '#808080',
        'aren_bad': '#000000',
        'aren_good': '#808080',
    }

    for i, (key, title, ylabel) in enumerate(metrics_cfg):
        zb_vals = [m.get(key, 0.0) for m in zhen_bad_metrics]
        zg_vals = [m.get(key, 0.0) for m in zhen_good_metrics]
        ab_vals = [m.get(key, 0.0) for m in aren_bad_metrics]
        ag_vals = [m.get(key, 0.0) for m in aren_good_metrics]
        ax = axes[i]
        # 均值高度
        zb_mean = sum(zb_vals) / len(zb_vals) if zb_vals else 0.0
        zg_mean = sum(zg_vals) / len(zg_vals) if zg_vals else 0.0
        ab_mean = sum(ab_vals) / len(ab_vals) if ab_vals else 0.0
        ag_mean = sum(ag_vals) / len(ag_vals) if ag_vals else 0.0

        # 柱子位置：同组相邻，不同组之间留空隙
        positions = [0, 1, 3, 4]
        heights = [zb_mean, zg_mean, ab_mean, ag_mean]
        bar_colors = [colors['zhen_bad'], colors['zhen_good'], colors['aren_bad'], colors['aren_good']]

        ax.bar(positions, heights, color=bar_colors, edgecolor='none', linewidth=0.6, alpha=0.95)
        ax.set_xticks(positions)
        ax.set_xticklabels(['易错词句子-zh', '非易错词句子-zh', '易错词句子-ar', '非易错词句子-ar'],fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.25)
        if fp:
            ax.set_title(title, fontproperties=fp)
            ax.set_ylabel(ylabel, fontproperties=fp)
        else:
            ax.set_title(title)
            ax.set_ylabel(ylabel)

        # 紧凑刻度：让 y 轴不从 0 起点，围绕四个柱子的范围微缩（不改标题）
        span = max(heights) - min(heights)
        if span == 0:
            # 所有柱等高时，给一个最小的上下缓冲，避免坐标轴奇异
            pad = max(0.70 * (max(heights) if heights else 1.0), 0.1)
        else:
            pad = 0.70 * span
        y_min = max(1e-6, min(heights) - pad)
        y_max = max(heights) + pad
        ax.set_ylim(y_min, y_max)

    # 总标题
    if fp:
        fig.suptitle('易错词句法复杂度四指标分组柱状图（zh-en、ar-en）', fontproperties=fp)
    else:
        fig.suptitle('易错词句法复杂度四指标分组柱状图（zh-en、ar-en）')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=300)
    if out_pdf:
        fig.savefig(out_pdf, format='pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='易错词句法复杂度分析实验')
    parser.add_argument('--corpus', default=os.path.join('zhen', 'train.ch'), help='zhen 源端语料库路径（中文）')
    parser.add_argument('--triggers2', default=os.path.join('zhen2', 'key_with_sent_all.json'), help='易错词 JSON (jsonlines) 路径（zhen2）')
    parser.add_argument('--triggers3', default=os.path.join('zhen3', 'key_with_sent_all.json'), help='易错词 JSON (jsonlines) 路径（zhen3）')
    parser.add_argument('--aren_corpus', default=os.path.join('aren', 'WikiMatrix.ar-en.ar'), help='aren 源端语料库路径（阿拉伯文）')
    parser.add_argument('--aren_triggers', default=os.path.join('aren2', 'key_with_sent_all.json'), help='易错词 JSON 路径（aren）')
    # parser.add_argument('--aren3_corpus', default=os.path.join('aren3', '99w.en'), help='aren3 源端语料库路径（英文）')
    parser.add_argument('--aren3_triggers', default=os.path.join('aren3', 'key_with_sent_all.json'), help='易错词 JSON 路径（并入 aren 的易错词索引）')
    parser.add_argument('--sample_size', type=int, default=2000, help='每个集合的样本数量（bad/good 各自）')
    parser.add_argument('--outdir', default=os.path.join('analysis_results'), help='输出目录')
    parser.add_argument('--pdfdir', default=os.path.join('pdf'), help='PDF 输出目录（已不再生成）')
    parser.add_argument('--stanza_dir', default=None, help='本地 Stanza 资源目录 (若不设置则使用 STANZA_RESOURCES_DIR 或默认路径)')
    args = parser.parse_args()

    # Load zhen data
    corpus_lines = readline(args.corpus)
    trigger_items2 = jsonreadline(args.triggers2) or []
    trigger_items3 = jsonreadline(args.triggers3) or []
    trigger_items = trigger_items2 + trigger_items3

    # 使用 index_list 直接选取集合（zhen2 与 zhen3 的并集）
    bad_indices_set = set()
    for item in trigger_items:
        idxs = item.get('index_list')
        if isinstance(idxs, list):
            for i in idxs:
                if isinstance(i, (int, float)):
                    bad_indices_set.add(int(i))
                elif isinstance(i, list) and i:
                    # 有些 JSON 可能形如 [index, score]
                    try:
                        bad_indices_set.add(int(i[0]))
                    except Exception:
                        pass
    zhen_bad_indices = list(bad_indices_set)
    zhen_bad_sample, zhen_good_sample = select_sets_by_index(corpus_lines, zhen_bad_indices, args.sample_size)

    # Load aren data（仅处理 aren，同时保留 aren3 的参数与索引对称性）
    aren_lines = readline(args.aren_corpus)
    aren_items = jsonreadline(args.aren_triggers) or []
    # aren3 不单独计算指标，只用于合并易错词索引
    # aren3_lines = readline(args.aren3_corpus)
    ar3_items = jsonreadline(args.aren3_triggers) or []
    aren_bad_set = set()
    for item in (aren_items + ar3_items):
        idxs = item.get('index_list')
        if isinstance(idxs, list):
            for i in idxs:
                if isinstance(i, (int, float)):
                    aren_bad_set.add(int(i))
                elif isinstance(i, list) and i:
                    try:
                        aren_bad_set.add(int(i[0]))
                    except Exception:
                        pass
    aren_bad_indices = list(aren_bad_set)
    aren_bad_sample, aren_good_sample = select_sets_by_index(aren_lines, aren_bad_indices, args.sample_size)

    # aren3 indices（不再单独生成样本，已合并到 aren_bad_set）
    # aren3_bad_set = set()
    # for item in ar3_items:
    #     idxs = item.get('index_list')
    #     if isinstance(idxs, list):
    #         for i in idxs:
    #             if isinstance(i, (int, float)):
    #                 aren3_bad_set.add(int(i))
    #             elif isinstance(i, list) and i:
    #                 try:
    #                     aren3_bad_set.add(int(i[0]))
    #                 except Exception:
    #                     pass
    # aren3_bad_indices = list(aren3_bad_set)
    # aren3_bad_sample, aren3_good_sample = select_sets_by_index(aren3_lines, aren3_bad_indices, args.sample_size)

    # Prepare nlp pipelines
    nlp_zh = ensure_stanza_zh(args.stanza_dir)
    nlp_en = ensure_stanza_en(args.stanza_dir)
    nlp_ar = ensure_stanza_ar(args.stanza_dir)

    # Compute metrics
    zhen_bad_metrics = compute_metrics_for_texts(zhen_bad_sample, nlp_zh)
    zhen_good_metrics = compute_metrics_for_texts(zhen_good_sample, nlp_zh)

    # 计算 aren（阿文）
    aren_bad_metrics = compute_metrics_for_texts(aren_bad_sample, nlp_ar)
    aren_good_metrics = compute_metrics_for_texts(aren_good_sample, nlp_ar)

    # 计算 aren3（英文）——移除单独计算，保持两语向：zhen、aren
    # aren3_bad_metrics = compute_metrics_for_texts(aren3_bad_sample, nlp_en)
    # aren3_good_metrics = compute_metrics_for_texts(aren3_good_sample, nlp_en)

    # Save metrics CSV（三个集合依次附加写入）
    csv_path = os.path.join(args.outdir, 'syntactic_complexity_metrics.csv')
    save_csv(csv_path, zhen_bad_metrics, 'Set_zhen_bad')
    save_csv(csv_path, zhen_good_metrics, 'Set_zhen_good')
    save_csv(csv_path, aren_bad_metrics, 'Set_aren_bad')
    save_csv(csv_path, aren_good_metrics, 'Set_aren_good')
    # save_csv(csv_path, aren3_bad_metrics, 'Set_aren3_bad')
    # save_csv(csv_path, aren3_good_metrics, 'Set_aren3_good')

    # Statistical tests（分别对 zhen 和 aren 的 bad/good 做检验）
    stats_zhen = run_tests(zhen_bad_metrics, zhen_good_metrics)
    stats_aren = run_tests(aren_bad_metrics, aren_good_metrics)
    # stats_aren3 = run_tests(aren3_bad_metrics, aren3_good_metrics)

    # 保存各自的统计结果
    stats_zhen_path = os.path.join(args.outdir, 'syntactic_complexity_stats_zhen.txt')
    stats_aren_path = os.path.join(args.outdir, 'syntactic_complexity_stats_aren.txt')
    # stats_aren3_path = os.path.join(args.outdir, 'syntactic_complexity_stats_aren3.txt')
    save_stats_txt(stats_zhen_path, stats_zhen)
    save_stats_txt(stats_aren_path, stats_aren)
    # save_stats_txt(stats_aren3_path, stats_aren3)

    # 控制台打印
    name_map = {
        'avg_dependency_depth': '依存深度 Avg Dependency Depth',
        'mean_sentence_length': '平均句长 Mean Sentence Length',
        'avg_dependency_distance': '依存弧跨度 Avg Dependency Distance',
        'subordinate_clause_ratio': '从句比例 Subordinate Clause Ratio',
    }
    print('\n统计检验（Welch t 检验 与 Mann–Whitney U）结果：')
    print('【zhen 组】')
    for k in ['avg_dependency_depth','mean_sentence_length','avg_dependency_distance','subordinate_clause_ratio']:
        v = stats_zhen.get(k, {})
        t_p = v.get('t_p', float('nan'))
        star = ''
        try:
            if t_p < 0.001:
                star = '***'
            elif t_p < 0.01:
                star = '**'
            elif t_p < 0.05:
                star = '*'
        except TypeError:
            star = ''
        print(f"- {name_map.get(k,k)}: t_stat={v.get('t_stat'):.4f}, t_p={t_p:.6f} {star}; u_stat={v.get('u_stat'):.4f}, u_p={v.get('u_p'):.6f}")
    print('【aren 组】')
    for k in ['avg_dependency_depth','mean_sentence_length','avg_dependency_distance','subordinate_clause_ratio']:
        v = stats_aren.get(k, {})
        t_p = v.get('t_p', float('nan'))
        star = ''
        try:
            if t_p < 0.001:
                star = '***'
            elif t_p < 0.01:
                star = '**'
            elif t_p < 0.05:
                star = '*'
        except TypeError:
            star = ''
        print(f"- {name_map.get(k,k)}: t_stat={v.get('t_stat'):.4f}, t_p={t_p:.6f} {star}; u_stat={v.get('u_stat'):.4f}, u_p={v.get('u_p'):.6f}")
    # 移除 aren3 的打印块，保持只输出 zhen/aren 两语向
    # print('【aren3 组】')
    # for k in ['avg_dependency_depth','mean_sentence_length','avg_dependency_distance','subordinate_clause_ratio']:
    #     v = stats_aren3.get(k, {})
    #     t_p = v.get('t_p', float('nan'))
    #     star = ''
    #     try:
    #         if t_p < 0.001:
    #             star = '***'
    #         elif t_p < 0.01:
    #             star = '**'
    #         elif t_p < 0.05:
    #             star = '*'
    #     except TypeError:
    #         star = ''
    #     print(f"- {name_map.get(k,k)}: t_stat={v.get('t_stat'):.4f}, t_p={t_p:.6f} {star}; u_stat={v.get('u_stat'):.4f}, u_p={v.get('u_p'):.6f}")

    # 绘图：每个子图绘制四个柱子（zhen_bad/zhen_good 与 aren_bad/aren_good）
    plot_all_metrics_grid(
        zhen_bad_metrics,
        zhen_good_metrics,
        aren_bad_metrics,
        aren_good_metrics,
        out_png=os.path.join(args.outdir, 'barchart_all_metrics.png'),
        out_pdf=None,
    )

    print('\n完成：')
    print(f"CSV 指标已保存到: {csv_path}")
    print(f"zhen 统计检验结果已保存到: {stats_zhen_path}")
    print(f"aren 统计检验结果已保存到: {stats_aren_path}")
    # print(f"aren3 统计检验结果已保存到: {stats_aren3_path}")
    print(f"合并柱状图已保存到: {os.path.join(args.outdir, 'barchart_all_metrics.png')}")


if __name__ == '__main__':
    main()