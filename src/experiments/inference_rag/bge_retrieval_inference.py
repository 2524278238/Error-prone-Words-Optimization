import jieba
import random
import os
import argparse
import json
import re
import torch
import numpy as np
from src.utils.common import *
import sacrebleu
from FlagEmbedding import FlagModel

# 配置参数
parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str, default='aren', help='Language pair: aren, zhen, ende')
parser.add_argument('--model_ver', type=str, default='llama3', help='Model version: llama2, llama3')
parser.add_argument('--gpu', type=str, default='0', help='CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

# 设置 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# 路径映射配置 (与 run_tgt.py 保持一致)
CONFIG = {
    'aren': {
        'train_src': '/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.ar',
        'train_ref': '/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.en',
        'test_ref': '/public/home/xiangyuduan/lyt/basedata/aren/test_ref.en',
        'prompt_template': 'Translate Arabic to English:\nArabic: {src}\nEnglish: {tgt}\n\n',
        'query_template': 'Translate Arabic to English:\nArabic: {src}\nEnglish:'
    },
    'zhen': {
        'train_src': '/public/home/xiangyuduan/lyt/basedata/125/train.ch',
        'train_ref': '/public/home/xiangyuduan/lyt/basedata/125/train.en',
        'test_ref': '/public/home/xiangyuduan/lyt/basedata/zhen/test_ref.en',
        'prompt_template': 'Translate Chinese to English:\nChinese: {src}\nEnglish: {tgt}\n\n',
        'query_template': 'Translate Chinese to English:\nChinese: {src}\nEnglish:'
    },
    'ende': {
        'train_src': '/public/home/xiangyuduan/lyt/basedata/459/train.en',
        'train_ref': '/public/home/xiangyuduan/lyt/basedata/459/train.de',
        'test_ref': '/public/home/xiangyuduan/lyt/basedata/ende/test_ref.de',
        'prompt_template': 'Translate English to German:\nEnglish: {src}\nGerman: {tgt}\n\n',
        'query_template': 'Translate English to German:\nEnglish: {src}\nGerman:'
    }
}

# 模型路径
MODELS = {
    'llama2': '/public/home/xiangyuduan/models/hf/Llama-2-7b-hf',
    'llama3': '/public/home/xiangyuduan/models/hf/Llama-3.1-8B'
}

# BGE 模型路径
BGE_MODEL_PATH = '/public/home/xiangyuduan/lyt/model/bge-m3'

# 1. 加载基础数据
print(f"Loading data for {args.lang} ...")
cfg = CONFIG[args.lang]
train_src = readline(cfg['train_src'])
train_ref = readline(cfg['train_ref'])
test_ref = readline(cfg['test_ref'])

# 确定数据目录
data_dir_name = f"{args.lang}{'2' if args.model_ver == 'llama2' else '3'}"
base_dir = f"/public/home/xiangyuduan/lyt/bad_word/key_data/{data_dir_name}"

# 输入文件：testbase_{model}_tgt.json
input_file_name = f"testbase_{args.model_ver}_tgt.json"
input_path = os.path.join(base_dir, input_file_name)

if not os.path.exists(input_path):
    print(f"Error: Input file not found: {input_path}")
    local_path = os.path.join(os.getcwd(), 'key_data', data_dir_name, input_file_name)
    if os.path.exists(local_path):
        print(f"Found local file: {local_path}")
        input_path = local_path
        base_dir = os.path.dirname(local_path)
    else:
        print("Cannot find input file. Exiting.")
        exit(1)

data = jsonreadline(input_path)

# 加载 BGE 模型
print("Loading BGE model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    bge_model = FlagModel(BGE_MODEL_PATH, device=device)
except Exception as e:
    print(f"Failed to load BGE model: {e}")
    exit(1)

# 加载易错词 COMET 数据
# 尝试加载带 COMET 分数的索引文件
comet_data_path = os.path.join(base_dir, "tgt_key_with_sent_comet.json")
has_comet_data = False
comet_map = {}

if os.path.exists(comet_data_path):
    print(f"Loading COMET data from {comet_data_path}...")
    comet_data_list = jsonreadline(comet_data_path)
    comet_map = {item['trigger_word']: item for item in comet_data_list}
    has_comet_data = True
else:
    print(f"Warning: {comet_data_path} not found.")
    # 尝试加载普通索引文件作为降级
    normal_cfc_path = os.path.join(base_dir, "tgt_key_with_sent.json")
    if os.path.exists(normal_cfc_path):
        print(f"Loading index data from {normal_cfc_path} (without COMET scores)...")
        cfc_data_list = jsonreadline(normal_cfc_path)
        comet_map = {item['trigger_word']: item for item in cfc_data_list}
    else:
        print(f"Error: No index file found at {normal_cfc_path}")
        exit(1)

baseline_mt = [i.get('test', '') if i.get('test') else i.get('mt', '') for i in data]

# 2. 加载 LLM 模型
model_path = MODELS[args.model_ver]
print(f"Loading LLM model from {model_path} ...")
try:
    tokenizer, model = load_vLLM_model(model_path, seed=0, tensor_parallel_size=1, half_precision=True)
except Exception as e:
    print(f"Failed to load LLM model: {e}")
    print("If you are on Windows, vLLM is not supported. Please run this script on Linux with vLLM installed.")
    exit(1)

# 辅助函数
def random_pro():
    i = random.randint(0, len(train_src)-1)
    return cfg['prompt_template'].format(src=train_src[i], tgt=train_ref[i])

def normalize(arr):
    arr = np.array(arr)
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())

# 3. 推理主循环
test_srcx = []
test_nmtx = []
test_refx = []

# 输出路径
output_filename = f"{args.lang}_{args.model_ver}_tgt_infer_bge.json"
savepath = os.path.join(base_dir, output_filename)
print(f"Results will be saved to {savepath}")

processed_data = []

print("Processing samples...")
for i in range(len(data)):
    item = data[i]
    trigger_word = item.get('trigger_word')
    src = item['src']
    
    mt_result = ""
    best_src = ""
    best_ref = ""
    
    if trigger_word and trigger_word in comet_map:
        # print(f"Trigger: {trigger_word}")
        cfc_item = comet_map[trigger_word]
        
        # 获取候选
        if 'index_list' in cfc_item and len(cfc_item['index_list']) > 0:
            # 兼容两种格式：
            # 1. [[index, score], ...] (from comet file)
            # 2. [index, index, ...] (from normal file)
            
            first_idx = cfc_item['index_list'][0]
            if isinstance(first_idx, list) or isinstance(first_idx, tuple):
                indices = [x[0] for x in cfc_item['index_list']]
                comet_scores = [x[1] for x in cfc_item['index_list']]
            else:
                indices = cfc_item['index_list']
                comet_scores = [0.0] * len(indices) # 无分数
                
            # 限制候选数量，避免计算过慢
            indices = indices[:50] 
            comet_scores = comet_scores[:50]
            
            # 获取候选源文和译文
            # 注意：这里的 indices 对应的是 train_src/train_ref
            candidate_srcs = []
            candidate_refs = []
            valid_indices = []
            valid_comet_scores = []
            
            for k, idx in enumerate(indices):
                try:
                    s = train_src[idx]
                    t = train_ref[idx]
                    candidate_srcs.append(s)
                    candidate_refs.append(t)
                    valid_indices.append(idx)
                    valid_comet_scores.append(comet_scores[k])
                except IndexError:
                    continue
            
            if candidate_srcs:
                # 计算 BGE 相似度 (Source to Source)
                # 这里的 src 是当前测试句子的 source
                # candidate_srcs 是包含 trigger_word (在 Target 端) 的训练句子的 source
                query_emb = bge_model.encode(src)
                corpus_embs = bge_model.encode(candidate_srcs)
                
                bge_scores = torch.nn.functional.cosine_similarity(
                    torch.tensor(query_emb).unsqueeze(0),
                    torch.tensor(corpus_embs),
                    dim=1
                ).numpy()
                
                # 归一化并结合
                if has_comet_data and any(valid_comet_scores):
                    comet_norm = normalize(valid_comet_scores)
                    bge_norm = normalize(bge_scores)
                    qz = 0.1 # COMET 权重
                    combined = comet_norm * qz + bge_norm * (1 - qz)
                else:
                    # 只有 BGE
                    combined = bge_scores
                
                best_idx_in_list = np.argmax(combined)
                
                best_src = candidate_srcs[best_idx_in_list]
                best_ref = candidate_refs[best_idx_in_list]
                
                # 构造 Prompt
                prompt_content = cfg['prompt_template'].format(src=best_src, tgt=best_ref)
                
                full_prompt = prompt_content + cfg['query_template'].format(src=src)
                
                output = generate_with_vLLM_model(model, [full_prompt], n=1, stop=['\n'], top_p=0.00001, top_k=1, max_tokens=150, temperature=0)
                mt_result = output[0].outputs[0].text.strip()
                
                item['prompt'] = [prompt_content]
                item['retrieved_src'] = best_src
                item['retrieved_ref'] = best_ref
            else:
                 # 候选为空
                 mt_result = baseline_mt[i]
        else:
            # 无索引
            mt_result = baseline_mt[i]
            
    else:
        # 不触发时，使用 baseline 结果
        mt_result = baseline_mt[i]
        
    item['new_mt'] = mt_result
    
    test_srcx.append(src)
    test_nmtx.append(mt_result)
    test_refx.append(test_ref[i])
    
    processed_data.append(item)

del model
import torch
torch.cuda.empty_cache()

# 4. 评估
print("Calculating metrics...")
try:
    comet_scores, avg_comet = count_comet(test_srcx, test_refx, test_nmtx)
    
    try:
        cometfree_scores, avg_cometfree = count_comet(test_srcx, test_refx, test_nmtx, model_path='/public/home/xiangyuduan/models/hf/wmt22-cometkiwi-da/checkpoints/model.ckpt')
        print(f"COMET: {avg_comet}")
        print(f"COMET-Kiwi: {avg_cometfree}")
    except Exception as e:
        print(f"Skipping COMET-Kiwi: {e}")
        cometfree_scores = [0.0] * len(test_srcx)

except Exception as e:
    print(f"Error calculating COMET: {e}")
    comet_scores = [0.0] * len(test_srcx)
    cometfree_scores = [0.0] * len(test_srcx)

bleu = sacrebleu.corpus_bleu(test_nmtx, [test_refx]).score
print(f"BLEU: {bleu}")

# 5. 回填分数并保存
print(f"Saving final results to {savepath} ...")
with open(savepath, 'w', encoding='utf-8') as f:
    for i, item in enumerate(processed_data):
        item['comet'] = float(comet_scores[i]) if isinstance(comet_scores, list) else 0.0
        item['comet_free'] = float(cometfree_scores[i]) if isinstance(cometfree_scores, list) else 0.0
        item['mt'] = test_nmtx[i]
        
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

print("Done.")
