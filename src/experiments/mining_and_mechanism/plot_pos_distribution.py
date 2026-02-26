
import json
import matplotlib.pyplot as plt
from collections import Counter
import os

# Language setting: 'en' for English, 'zh' for Chinese
LANGUAGE = 'zh'  # Change this to 'zh' for Chinese labels

# Set fonts based on language
if LANGUAGE == 'zh':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
else:
    plt.rcParams['font.sans-serif'] = ['Arial', 'Times New Roman', 'Calibri', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    "font.size": 14,        # 默认字体大小
    "xtick.labelsize": 12,  # x 轴刻度
    "ytick.labelsize": 12,  # y 轴刻度
    "legend.fontsize": 14,  # 图例
    "axes.titlesize": 18,   # 标题字体大小
    "axes.labelsize": 14,   # 轴标签字体大小
})

# Mapping from Chinese POS categories to English for plotting
EN_CATEGORY_MAP = {
    '名词': 'Noun',
    '动词': 'Verb',
    '形容词': 'Adjective',
    '副词': 'Adverb',
    '代词': 'Pronoun',
    '介词': 'Preposition',
    '连词': 'Conjunction',
    '数词': 'Numeral',
    '量词': 'Classifier',
    '助词': 'Particle',
    '符号': 'Symbol',
    '语气词': 'Modal',
    '术语': 'Terminology',
    '非源端词': 'Non-source word',
    '多义词': 'Polysemous'
}

# Language-specific labels
LABELS = {
    'en': {
        'title': 'POS Distribution of Error-Prone Words',
        'ylabel': 'Percentage (%)',
        'xlabel': 'POS Category',
        'lang_pairs': {
            'Chinese-to-English': 'Chinese-to-English',
            'Arabic-to-English': 'Arabic-to-English'
        },
        'models': {
            'llama2-7B': 'llama2-7B',
            'llama3.1-8B': 'llama3.1-8B'
        }
    },
    'zh': {
        'title': '易错词词性分布',
        'ylabel': '百分比 (%)',
        'xlabel': '词性类别',
        'lang_pairs': {
            'Chinese-to-English': '中译英',
            'Arabic-to-English': '阿译英'
        },
        'models': {
            'llama2-7B': 'llama2-7B',
            'llama3.1-8B': 'llama3.1-8B'
        }
    }
}

def load_pos_data(filepath):
    """
    Reads a JSONL file and returns POS category counts.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f'Error: File not found at {filepath}')
        return None

    # Handle empty files
    if not lines:
        print(f'Warning: File is empty, skipping: {filepath}')
        return None

    categories = [json.loads(line)['category'] for line in lines]
    category_counts = Counter(categories)
    return category_counts

def generate_combined_pos_chart():
    """
    Generates a combined 2x2 subplot chart for all POS distributions.
    """
    # Define all combinations
    combinations = [
        {
            'path': 'zhen3/zhen3_very_pos_tags_fixed.jsonl',
            'title': 'Zh->En, Llama3',
            'lang_pair': 'Chinese-to-English',
            'model': 'llama3.1-8B'
        },
        {
            'path': 'zhen2/zhen2_very_pos_tags_fixed.jsonl',
            'title': 'Zh->En, Llama2',
            'lang_pair': 'Chinese-to-English',
            'model': 'llama2-7B'
        },
        {
            'path': 'aren3/aren3_very_pos_tags_fixed.jsonl',
            'title': 'Ar->En, Llama3',
            'lang_pair': 'Arabic-to-English',
            'model': 'llama3.1-8B'
        },
        {
            'path': 'aren2/aren2_very_pos_tags_fixed.jsonl',
            'title': 'Ar->En, Llama2',
            'lang_pair': 'Arabic-to-English',
            'model': 'llama2-7B'
        }
    ]
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(LABELS[LANGUAGE]['title'], 
                 fontsize=16, y=0.98)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for idx, combo in enumerate(combinations):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Load data
        input_filepath = os.path.join(base_dir, combo['path'])
        category_counts = load_pos_data(input_filepath)
        
        if category_counts is None:
            ax.text(0.5, 0.5, f'Error loading data for {combo["title"]}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{combo["title"]} (Error)')
            continue
        
        # Sort categories by count for better visualization
        sorted_categories = sorted(category_counts.items(), key=lambda item: item[1], reverse=True)
        labels = [item[0] for item in sorted_categories]
        counts = [item[1] for item in sorted_categories]
        
        # Choose labels based on language setting
        if LANGUAGE == 'zh':
            plot_labels = labels  # Use original Chinese labels
        else:
            plot_labels = [EN_CATEGORY_MAP.get(label, label) for label in labels]
        
        # Calculate percentages
        total_count = sum(counts)
        percentages = [(count / total_count) * 100 for count in counts]

        # Print the distribution results for analysis
        print(f"\n--- Distribution for: {combo['title']} ---")
        for (label, count), percentage in zip(sorted_categories, percentages):
            print(f"Category: {label:<15} | Count: {count:<5} | Percentage: {percentage:.2f}%")
        print(f"--- Total: {total_count} ---\n")

        # Create bar chart
        bars = ax.bar(plot_labels, percentages, color='gray', edgecolor='black', linewidth=1.2, alpha=0.7)
        
        # Add percentage labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}%', 
                   va='bottom', ha='center', fontsize=12)

        #        #ax.set_xlabel('POS Category')
        ax.set_ylabel(LABELS[LANGUAGE]['ylabel'])
        lang_pair_label = LABELS[LANGUAGE]['lang_pairs'][combo["lang_pair"]]
        model_label = LABELS[LANGUAGE]['models'][combo["model"]]
        ax.set_title(f'{lang_pair_label}, {model_label}')
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust subplot spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for main title
    
    # Save the combined figure
    output_filename = f'all_models_pos_distribution_{LANGUAGE}.png'
    output_filepath = os.path.join(base_dir, output_filename)
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Successfully generated combined POS chart ({LANGUAGE}): {output_filepath}')

if __name__ == '__main__':
    # Generate combined POS distribution chart
    generate_combined_pos_chart()
