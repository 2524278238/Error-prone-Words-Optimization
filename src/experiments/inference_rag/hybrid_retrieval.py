from FlagEmbedding import FlagModel
from src.utils.common import *
import torch
import json
import sacrebleu
import numpy as np

# 1. 加载BGE模型
print("Loading BGE model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bge_model = FlagModel('/public/home/xiangyuduan/lyt/model/bge-m3', device=device)

# 2. 读取数据
print("Loading data...")
test_ref = readline('/public/home/xiangyuduan/lyt/basedata/aren/test_ref.en')
large_src = readline('/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.ar')
large_ref = readline('/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.en')
data = jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/aren2/aren_llama2.json')
key_data = jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/aren2/key_with_sent.json')
baseline_mt = [i['test'] for i in jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/aren2/testbase_llama2.json')]
comet_data = jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/aren2/key_with_sent_comet.json')

# 加载llama3模型
print("Loading Llama model...")
#model_path = '/public/home/xiangyuduan/bli/blidata/models/hf/Llama-3.1-8B'
model_path='/public/home/xiangyuduan/bli/blidata/models/hf/Llama-2-7b-hf'
tokenizer, llm_model = load_vLLM_model(model_path, seed=42, tensor_parallel_size=1, half_precision=True)

save_path = '/public/home/xiangyuduan/lyt/bad_word/key_data/aren2/testllama2_comet_bge.json'

# 构建trigger_word到comet_data的映射
comet_map = {item['trigger_word']: item for item in comet_data}

results = []
test_nmt = []
test_src = []

# 归一化函数
def normalize(arr):
    arr = np.array(arr)
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())

def find_cfc(word, data):
    for i in data:
        if i['trigger_word'] == word:
            return i

def find_mincfc(dic, data):
    avg_comet = 1
    mincfc = ''
    cfc_list = dic.keys()
    for i in cfc_list:
        a = find_cfc(i, data)['avg_comet']
        if a < avg_comet:
            mincfc = i
            avg_comet = a
    return mincfc

print("Processing samples...")
for i, item in enumerate(data):
    src = item['src']
    test_src.append(src)
    trigger_word = item.get('trigger_word', None)
    print(f"Processing {i+1}/{len(data)}: {trigger_word}")

    if trigger_word:
        # 选择最难cfc
        #tri = find_mincfc(trigger_word, key_data)
        tri = trigger_word
        if tri in comet_map:
            comet_item = comet_map[tri]
            # comet_item['index_list']: [[index, comet_score], ...]
            indices = [x[0] for x in comet_item['index_list']]
            comet_scores = [x[1] for x in comet_item['index_list']]
            # 获取候选源文
            candidate_srcs = [large_src[idx] for idx in indices]
            candidate_refs = [large_ref[idx] for idx in indices]
            # 计算BGE相似度
            query_emb = bge_model.encode(src)
            corpus_embs = bge_model.encode(candidate_srcs)
            bge_scores = torch.nn.functional.cosine_similarity(
                torch.tensor(query_emb).unsqueeze(0),
                torch.tensor(corpus_embs),
                dim=1
            ).numpy()
            # 归一化
            comet_norm = normalize(comet_scores)
            bge_norm = normalize(bge_scores)
            # 综合分数（可调整权重）
            qz=0.75
            combined = comet_norm * qz + bge_norm * (1 - qz)
            best_idx = np.argmax(combined)
            best_index = indices[best_idx]
            best_src = candidate_srcs[best_idx]
            best_ref = candidate_refs[best_idx]
            # 构造prompt
            prompt = f"Translate Arabic to English:\nArabic: {best_src}\nEnglish: {best_ref}\nTranslate Arabic to English:\nArabic: {src}\nEnglish:"
            # 生成翻译
            output = generate_with_vLLM_model(llm_model, prompt, n=1, stop=['\n'], top_p=0.00001, top_k=1, max_tokens=150, temperature=0)
            mt = output[0].outputs[0].text.strip()
        else:
            print(f"Warning: No comet data found for sent_id {i}, using baseline")
            mt = baseline_mt[i]
            best_src = ""
            best_ref = ""
    else:
        # 非触发词句子使用baseline翻译
        mt = baseline_mt[i]
        best_src = ""
        best_ref = ""

    # 保存结果
    result = {
        'sent_id': i,
        'src': src,
        'trigger_word': trigger_word,
        'retrieved_src': best_src,
        'retrieved_ref': best_ref,
        'new_mt': mt
    }
    results.append(result)
    test_nmt.append(mt)

# 释放模型
del llm_model

def safe_count_comet(test_src, test_ref, test_nmt):
    try:
        _, comet = count_comet(test_src, test_ref, test_nmt)
    except Exception as e:
        print(f"COMET error: {e}")
        comet = -1
    try:
        _, cometfree = count_comet(test_src, test_ref, test_nmt, model_path='/public/home/xiangyuduan/bli/blidata/models/hf/wmt22-cometkiwi-da/checkpoints/model.ckpt')
    except Exception as e:
        print(f"COMET-free error: {e}")
        cometfree = -1
    bleu = sacrebleu.corpus_bleu(test_nmt, [test_ref]).score
    return comet, cometfree, bleu

# 统一计算评分
print("Computing metrics...")
comet, cometfree, bleu = safe_count_comet(test_src, test_ref, test_nmt)
print(f"COMET: {comet}")
print(f"COMET-free: {cometfree}")
print(f"BLEU: {bleu}")

# 保存结果
print(f"Saving results to {save_path}...")
with open(save_path, 'w', encoding='utf-8') as f:
    for result in results:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')

print("Done!") 