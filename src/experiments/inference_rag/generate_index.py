import os
import json
import argparse
from comet import load_from_checkpoint
from src.utils.common import *
import torch

# 路径映射配置 (与 run_tgt.py 保持一致)
CONFIG = {
    'aren': {
        'train_src': '/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.ar',
        'train_ref': '/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.en',
    },
    'zhen': {
        'train_src': '/public/home/xiangyuduan/lyt/basedata/125/train.ch',
        'train_ref': '/public/home/xiangyuduan/lyt/basedata/125/train.en',
    },
    'ende': {
        'train_src': '/public/home/xiangyuduan/lyt/basedata/459/train.en',
        'train_ref': '/public/home/xiangyuduan/lyt/basedata/459/train.de',
    }
}

COMET_MODEL_PATH = '/public/home/xiangyuduan/models/hf/wmt22-cometkiwi-da/checkpoints/model.ckpt'

def process_lang(lang, version, gpu_id):
    print(f"\nProcessing {lang} {version}...")
    
    # 确定数据目录
    data_dir_name = f"{lang}{'2' if version == 'llama2' else '3'}"
    base_dir = f"/public/home/xiangyuduan/lyt/bad_word/key_data/{data_dir_name}"
    
    # 输入文件: tgt_key_with_sent.json
    input_path = os.path.join(base_dir, "tgt_key_with_sent.json")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return

    data = jsonreadline(input_path)
    
    # 加载训练数据 (用于获取句子内容)
    cfg = CONFIG[lang]
    print(f"Loading training data from {cfg['train_src']} and {cfg['train_ref']}...")
    train_src = readline(cfg['train_src'])
    train_ref = readline(cfg['train_ref'])
    
    # 加载 COMET 模型
    print(f"Loading COMET model from {COMET_MODEL_PATH}...")
    try:
        model = load_from_checkpoint(COMET_MODEL_PATH)
        # 手动设置设备，虽然 load_from_checkpoint 可能会尝试，但 predict 时指定 gpus 参数更直接
    except Exception as e:
        print(f"Failed to load COMET model: {e}")
        return

    # 准备计算 COMET 分数
    # 我们需要为每个易错词的 index_list 中的每个句子计算 QE 分数 (Source, Target)
    # 因为这里是 "tgt_key_with_sent"，我们假设它是为了检索高质量的 Parallel Data
    # 通常 COMET 需要 (Src, Ref, MT) 或者 QE 需要 (Src, MT)
    # 这里的场景是：我们想评价 (Src, Ref) 对的质量，或者说 Ref 作为 Tgt 的质量
    # wmt22-cometkiwi-da 是一个 QE 模型 (Reference-free)，它接受 (Src, MT) 并评分
    # 在这里，我们的 "MT" 就是训练集中的 Target (Ref)
    
    processed_count = 0
    updated_data = []
    
    # 收集所有需要打分的样本，批量处理以提高效率
    all_samples = [] # list of {"src": s, "mt": t}
    sample_indices = [] # list of (word_index, list_index_in_word) 记录回填位置
    
    print("Preparing samples for scoring...")
    for i, item in enumerate(data):
        indices = item.get('index_list', [])
        
        # item['index_list'] 可能是 [idx1, idx2, ...] 或 [[idx1, score], ...]
        # 我们这里假设输入是纯索引列表 [idx1, idx2, ...] (因为是生成 comet 索引)
        # 如果已经是带分数的，我们只取索引重新计算
        
        current_indices = []
        if indices and isinstance(indices[0], list):
             current_indices = [x[0] for x in indices]
        else:
             current_indices = indices
             
        # 限制数量，避免过多？这里还是全量计算比较好，或者限制每个词最多N个
        # 为了效率，这里假设不做额外限制，全量计算
        
        for k, idx in enumerate(current_indices):
            try:
                s = train_src[idx]
                t = train_ref[idx]
                all_samples.append({"src": s, "mt": t})
                sample_indices.append((i, k)) # 第 i 个词，第 k 个句子
            except IndexError:
                continue
        
        # 保持结构，准备更新
        item['index_list'] = current_indices # 暂时重置为纯索引，稍后回填为 [idx, score]
        updated_data.append(item)

    print(f"Total samples to score: {len(all_samples)}")
    
    if all_samples:
        # 批量预测
        # batch_size 可以适当调大
        batch_size = 64
        print(f"Predicting scores with batch_size={batch_size}...")
        model_output = model.predict(all_samples, batch_size=batch_size, gpus=1)
        scores = model_output.scores
        
        # 回填分数
        print("Updating data with scores...")
        
        # 临时存储： updated_data[i]['scores'] = {k: score}
        temp_scores = {} 
        
        for idx, score in zip(sample_indices, scores):
            word_idx, list_idx = idx
            if word_idx not in temp_scores:
                temp_scores[word_idx] = {}
            temp_scores[word_idx][list_idx] = float(score)
            
        # 构建最终结构
        for i, item in enumerate(updated_data):
            original_indices = item['index_list']
            new_index_list = []
            
            if i in temp_scores:
                word_scores = temp_scores[i]
                for k, idx_val in enumerate(original_indices):
                    if k in word_scores:
                        new_index_list.append([idx_val, word_scores[k]])
                    else:
                        # 理论上不应该发生，除非 IndexError
                        pass
                
                # 排序：按分数降序排列 (高质量在前)
                new_index_list.sort(key=lambda x: x[1], reverse=True)
                item['index_list'] = new_index_list
            else:
                # 该词没有任何有效句子
                item['index_list'] = []
                
    else:
        print("No samples found to score.")

    # 保存
    output_path = os.path.join(base_dir, "tgt_key_with_sent_comet.json")
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in updated_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
            
    print(f"Done processing {lang} {version}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # 定义所有任务
    tasks = [
        ('aren', 'llama2'),
        ('aren', 'llama3'),
        ('zhen', 'llama2'),
        ('zhen', 'llama3'),
        ('ende', 'llama2'),
        ('ende', 'llama3'),
    ]
    
    for lang, ver in tasks:
        process_lang(lang, ver, args.gpu)
