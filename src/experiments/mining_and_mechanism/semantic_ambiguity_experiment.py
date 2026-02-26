import os
import sys
import argparse
import csv
from typing import List, Dict, Tuple

# 项目工具
from mydef import jsonreadline

# 依赖懒加载（OpenHowNet、NLTK WordNet、SciPy）
OpenHowNet = None
hownet_dict = None
nltk = None
wn = None
stats = None


def ensure_openhownet():
    global OpenHowNet, hownet_dict
    # 若已初始化则直接返回，避免重复下载/初始化
    if hownet_dict is not None:
        return
    # 确保包已安装
    if OpenHowNet is None:
        try:
            import OpenHowNet as _OpenHowNet
            OpenHowNet = _OpenHowNet
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "OpenHowNet", "-q"])  # 安装 OpenHowNet
            import OpenHowNet as _OpenHowNet
            OpenHowNet = _OpenHowNet
    # 先尝试直接初始化字典
    try:
        hownet_dict = OpenHowNet.HowNetDict()
        return
    except Exception:
        pass
    # 若资源未准备好，下载后再初始化（只执行一次）
    try:
        if hasattr(OpenHowNet, 'download'):
            OpenHowNet.download()
        else:
            OpenHowNet.Download.download()
        hownet_dict = OpenHowNet.HowNetDict()
    except Exception:
        # 下载/初始化失败则保持 None，调用方会兜底计数为 0
        hownet_dict = None


def ensure_nltk_wordnet():
    global nltk, wn
    if nltk is None:
        try:
            import nltk as _nltk
            nltk = _nltk
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk", "-q"])  # 安装 nltk
            import nltk as _nltk
            nltk = _nltk
    try:
        from nltk.corpus import wordnet as _wn
        wn = _wn
        # 确保 WordNet 语料可用
        try:
            wn.ensure_loaded()
        except Exception:
            nltk.download('wordnet', quiet=True)
            wn.ensure_loaded()
        # 额外确保 Open Multilingual Wordnet 资源可用（用于阿语）
        try:
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            nltk.download('omw-1.4', quiet=True)
    except Exception:
        from nltk.corpus import wordnet as _wn
        wn = _wn


def ensure_scipy():
    global stats
    if stats is None:
        try:
            from scipy import stats as _stats
            stats = _stats
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "-q"])  # 安装 scipy
            from scipy import stats as _stats
            stats = _stats


# 义项数量（语义歧义度）计数缓存
sense_cache: Dict[Tuple[str, str], int] = {}


def count_senses(word: str, lang: str) -> int:
    """返回一个词的义项数量（语义歧义度）。lang in {'zh','en','ar'}；若未命中任何义项则默认计为 1"""
    key = (word, lang)
    if key in sense_cache:
        return sense_cache[key]
    try:
        if lang == 'zh':
            ensure_openhownet()
            senses = hownet_dict.get_sense(word)  # 返回 Sense 列表
            cnt = len(senses) if senses is not None else 0
        elif lang == 'en':
            ensure_nltk_wordnet()
            syns = wn.synsets(word)
            cnt = len(syns)
        elif lang == 'ar':
            ensure_nltk_wordnet()
            # 使用 Open Multilingual Wordnet 的阿语词条（语言码 'arb'）
            syns = wn.synsets(word, lang='arb')
            cnt = len(syns)
        else:
            cnt = 0
    except Exception:
        cnt = 0
    # 未查到义项时，默认至少为 1
    if cnt == 0:
        cnt = 1
    sense_cache[key] = cnt
    return cnt

# 新增：提取义项示例

def _safe_get(d: Dict, keys: List[str]):
    for k in keys:
        if k in d and d[k]:
            return d[k]
    return None


def get_sense_examples(word: str, lang: str, max_per_word: int = 3) -> List[str]:
    examples: List[str] = []
    try:
        if lang == 'zh':
            ensure_openhownet()
            senses = hownet_dict.get_sense(word) or []
            for s in senses[:max_per_word]:
                # 尝试获取定义、词性、义原等信息
                pos = _safe_get(s, ['pos', 'part_of_speech'])
                definition = _safe_get(s, ['definition', 'def', 'Def'])
                sememes = _safe_get(s, ['sememe', 'sememes'])
                if isinstance(sememes, list):
                    sememes_str = ','.join(map(str, sememes))
                else:
                    sememes_str = str(sememes) if sememes is not None else ''
                sense_id = _safe_get(s, ['sense_id', 'id'])
                examples.append(f"[{pos or ''}] {definition or ''} | sememes: {sememes_str} | sense_id: {sense_id or ''}")
        elif lang == 'ar':
            ensure_nltk_wordnet()
            syns = wn.synsets(word, lang='arb')
            for syn in syns[:max_per_word]:
                name = syn.name()
                pos = syn.pos()
                definition = syn.definition()
                # 阿语词条（如果可用）
                lemmas_ar = [l.name() for l in syn.lemmas(lang='arb')]
                extras = f" | lemmas(ar): {', '.join(lemmas_ar)}" if lemmas_ar else ''
                examples.append(f"{name} [{pos}]: {definition}{extras}")
        else:
            ensure_nltk_wordnet()
            syns = wn.synsets(word)
            for syn in syns[:max_per_word]:
                name = syn.name()
                pos = syn.pos()
                definition = syn.definition()
                exs = syn.examples()
                ex_part = f" examples: {'; '.join(exs)}" if exs else ''
                examples.append(f"{name} [{pos}]: {definition}{ex_part}")
    except Exception:
        pass
    return examples


def write_examples_file(bad_words: List[str], control_words: List[str], lang: str, out_dir: str, sample_size: int = 10, max_per_word: int = 3) -> str:
    os.makedirs(out_dir, exist_ok=True)
    # 选取有义项的词作为示例
    bad_sample = []
    control_sample = []
    for w in bad_words:
        cnt = count_senses(w, lang)
        if cnt > 0:
            bad_sample.append((w, cnt))
        if len(bad_sample) >= sample_size:
            break
    for w in control_words:
        cnt = count_senses(w, lang)
        if cnt > 0:
            control_sample.append((w, cnt))
        if len(control_sample) >= sample_size:
            break
    # 写文件
    out_path = os.path.join(out_dir, 'semantic_ambiguity_examples.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"语义歧义度示例（{lang}）\n\n")
        f.write("[易错词示例]\n")
        for w, cnt in bad_sample:
            f.write(f"- {w} (义项数: {cnt})\n")
            exs = get_sense_examples(w, lang, max_per_word=max_per_word)
            for i, e in enumerate(exs, 1):
                f.write(f"    {i}. {e}\n")
        f.write("\n[非易错词示例]\n")
        for w, cnt in control_sample:
            f.write(f"- {w} (义项数: {cnt})\n")
            exs = get_sense_examples(w, lang, max_per_word=max_per_word)
            for i, e in enumerate(exs, 1):
                f.write(f"    {i}. {e}\n")
    return out_path


def read_bad_words(zhen2_triggers_path: str) -> List[str]:
    """从 zhen2/key_with_sent_all.json 读取易错词集合（去重）。"""
    items = jsonreadline(zhen2_triggers_path) or []
    words = []
    for it in items:
        w = str(it.get('trigger_word', '')).strip()
        if w:
            words.append(w)
    # 去重保持稳定顺序
    seen = set()
    uniq = []
    for w in words:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


def read_control_words(zhen2_cfc_path: str, bad_set: set, score_threshold: float = 10.0, num_threshold: int = 0) -> List[str]:
    """从 zhen2/src_cfc_all.json 读取满足条件的高频非易错词，并排除易错词集合。
    过滤条件：score>score_threshold 且 num>num_threshold
    """
    items = jsonreadline(zhen2_cfc_path) or []
    words = []
    for it in items:
        try:
            score = float(it.get('score', 0))
        except Exception:
            score = 0.0
        try:
            num = int(it.get('num', 0))
        except Exception:
            num = 0
        if (score > score_threshold) and (num > num_threshold):
            w = str(it.get('trigger_word', '')).strip()
            if w and (w not in bad_set):
                words.append(w)
    # 去重保持稳定顺序
    seen = set()
    uniq = []
    for w in words:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


def compute_ambiguity_distribution(words: List[str], lang: str) -> List[int]:
    return [count_senses(w, lang) for w in words]


def save_stats_txt(path: str, stats_dict: Dict[str, float]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for k, v in stats_dict.items():
            f.write(f"{k}: {v}\n")


def plot_hist_and_cdf(bad_counts: List[int], good_counts: List[int], out_dir: str):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

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

    os.makedirs(out_dir, exist_ok=True)

    # 直方图
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    bins = range(0, max(bad_counts + good_counts) + 2)
    ax1.hist(bad_counts, bins=bins, alpha=0.7, color='#000000', label='易错词', density=False)
    ax1.hist(good_counts, bins=bins, alpha=0.5, color='#808080', label='非易错词', density=False)
    if fp:
        ax1.set_title('语义歧义度直方图（义项数量）', fontproperties=fp)
        ax1.set_xlabel('义项数量', fontproperties=fp)
        ax1.set_ylabel('词数', fontproperties=fp)
    else:
        ax1.set_title('语义歧义度直方图（义项数量）')
        ax1.set_xlabel('义项数量')
        ax1.set_ylabel('词数')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.25)
    hist_png = os.path.join(out_dir, 'semantic_ambiguity_hist.png')
    fig1.tight_layout()
    fig1.savefig(hist_png, dpi=300)
    plt.close(fig1)

    # CDF 曲线
    import numpy as np
    def ecdf(data):
        x = np.sort(np.array(data))
        y = np.arange(1, len(x) + 1) / len(x) if len(x) > 0 else np.array([])
        return x, y

    bx, by = ecdf(bad_counts)
    gx, gy = ecdf(good_counts)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.step(bx, by, where='post', color='#000000', label='易错词')
    ax2.step(gx, gy, where='post', color='#808080', label='非易错词')
    if fp:
        ax2.set_title('语义歧义度 CDF（义项数量）', fontproperties=fp)
        ax2.set_xlabel('义项数量', fontproperties=fp)
        ax2.set_ylabel('累计比例', fontproperties=fp)
    else:
        ax2.set_title('语义歧义度 CDF（义项数量）')
        ax2.set_xlabel('义项数量')
        ax2.set_ylabel('累计比例')
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.25)
    cdf_png = os.path.join(out_dir, 'semantic_ambiguity_cdf.png')
    fig2.tight_layout()
    fig2.savefig(cdf_png, dpi=300)
    plt.close(fig2)

    return hist_png, cdf_png


def main():
    parser = argparse.ArgumentParser(description='语义歧义度分析实验')
    parser.add_argument('--lang', choices=['zh', 'en', 'ar'], default='zh', help='词语语言（zh 中文使用 OpenHowNet；en 英文使用 WordNet；ar 阿文使用 OMW-1.4）')
    parser.add_argument('--bad_words_path', default=os.path.join('zhen3', 'key_with_sent_all.json'), help='易错词 JSON 路径（zhen2/key_with_sent_all.json）')
    parser.add_argument('--control_words_path', default=os.path.join('zhen3', 'src_cfc_all.json'), help='控制集 JSON 路径（zhen2/src_cfc_all.json）')
    parser.add_argument('--score_threshold', type=float, default=10.0, help='控制集筛选阈值：score>threshold')
    parser.add_argument('--num_threshold', type=int, default=0, help='控制集筛选阈值：num>threshold')
    parser.add_argument('--outdir', default=os.path.join('analysis_results'), help='输出目录')
    # 新增：示例输出控制
    parser.add_argument('--examples', type=int, default=10, help='每组输出的示例词数量（有义项的前N个）')
    parser.add_argument('--examples_per_word', type=int, default=3, help='每个词输出的义项示例数量')
    args = parser.parse_args()

    # 读取词集合
    bad_words = read_bad_words(args.bad_words_path)
    bad_set = set(bad_words)
    control_words = read_control_words(args.control_words_path, bad_set=bad_set, score_threshold=args.score_threshold, num_threshold=args.num_threshold)

    # 计算语义歧义度分布（义项数量）
    bad_counts = compute_ambiguity_distribution(bad_words, lang=args.lang)
    good_counts = compute_ambiguity_distribution(control_words, lang=args.lang)

    # 统计检验（Welch t-test）
    ensure_scipy()
    t_stat, t_p = stats.ttest_ind(bad_counts, good_counts, equal_var=False)
    out_stats = {
        'bad_mean': sum(bad_counts) / len(bad_counts) if bad_counts else 0.0,
        'good_mean': sum(good_counts) / len(good_counts) if good_counts else 0.0,
        't_stat': float(t_stat),
        't_p': float(t_p),
        'bad_size': len(bad_counts),
        'good_size': len(good_counts),
    }

    stats_path = os.path.join(args.outdir, 'semantic_ambiguity_stats.txt')
    save_stats_txt(stats_path, out_stats)

    # 可视化
    hist_png, cdf_png = plot_hist_and_cdf(bad_counts, good_counts, out_dir=args.outdir)

    # 新增：生成义项示例文件
    examples_path = write_examples_file(bad_words, control_words, lang=args.lang, out_dir=args.outdir, sample_size=args.examples, max_per_word=args.examples_per_word)

    # 控制台输出
    print('\n语义歧义度分析完成：')
    print(f"- 易错词数量: {len(bad_words)}，控制词数量: {len(control_words)}")
    print(f"- 义项数量均值：易错词={out_stats['bad_mean']:.3f}，非易错词={out_stats['good_mean']:.3f}")
    print(f"- Welch t 检验：t_stat={out_stats['t_stat']:.4f}，p={out_stats['t_p']:.6f}")
    print(f"- 统计结果保存: {stats_path}")
    print(f"- 直方图: {hist_png}")
    print(f"- CDF 图: {cdf_png}")
    print(f"- 义项示例: {examples_path}")


if __name__ == '__main__':
    main()