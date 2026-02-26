from FlagEmbedding import FlagModel
from src.utils.common import *
import torch
import json
import sacrebleu

# 1. 加载BGE模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FlagModel('/public/home/xiangyuduan/lyt/model/bge-m3', device=device)

# 2. 读取数据
data=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/aren2/aren_llama2.json')
baseline_mt=[i['test'] for i in data]
test_ref=readline('/public/home/xiangyuduan/lyt/basedata/aren/test_ref.en')
test_src=[i['src'] for i in data]


large_src = readline('/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.ar')
large_ref = readline('/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.en')

key_data = jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/aren2/key_with_sent.json')


# 加载llama2模型
model_path = '/public/home/xiangyuduan/bli/blidata/models/hf/Llama-2-7b-hf'
#model_path = '/public/home/xiangyuduan/bli/blidata/models/hf/Llama-3.1-8B'
tokenizer, llm_model = load_vLLM_model(model_path, seed=42, tensor_parallel_size=1, half_precision=True)

def get_src(t):
    return re.findall(f"\nChinese:(.*)\n", t)[0]
def find_cfc(word,data):
    for i in data:
        if i['trigger_word']==word:
            return i
def find_mincfc(dic,data):
    avg_comet=1
    mincfc=''
    cfc_list=dic.keys()
    for i in cfc_list:
        a=find_cfc(i,data)['avg_comet']
        if a<avg_comet:
            mincfc=i
            avg_comet=a
    return mincfc
results = []
test_nmt = []
test_src = []

for i, item in enumerate(data):
    print(f"Processing {i}/{len(data)}")
    test_src.append(item['src'])
    
    if item['trigger_word']:
        # 从key_data获取30个候选句子
        #tri=find_mincfc(item['trigger_word'],key_data)
        tri=item['trigger_word']
        candidates = []
        for key_item in key_data:
            if key_item['trigger_word'] == tri:
                candidates = [(large_src[idx], large_ref[idx]) for idx in key_item['index_list']]
                break
        
        # 使用BGE计算相似度
        query_embedding = model.encode(item['src'])
        
        corpus_embeddings = model.encode([src for src, _ in candidates])
        
        # 计算余弦相似度
        scores = torch.nn.functional.cosine_similarity(
            torch.tensor(query_embedding).unsqueeze(0),
            torch.tensor(corpus_embeddings),
            dim=1
        )
        
        # 获取最相似句子的索引
        best_idx = scores.argmax().item()
        
        # 构造prompt
        best_src, best_ref = candidates[best_idx]
        prompt = f"Translate Arabic to English:\nArabic: {best_src}\nEnglish: {best_ref}\nTranslate Arabic to English:\nArabic: {item['src']}\nEnglish:"
        
        # 生成翻译
        output = generate_with_vLLM_model(llm_model,prompt, n=1, stop=['\n'], top_p=0.00001, top_k=1, max_tokens=150, temperature=0)
        new_mt = output[0].outputs[0].text.strip()
        # 保存结果
        result = {
            'sent_id': item['sent_id'],
            'src': item['src'],
            'trigger_word': tri,
            'retrieved_src': best_src,
            'retrieved_ref': best_ref,
            'new_mt': new_mt
        }
    else:
        # 非触发词句子使用baseline翻译
        new_mt = baseline_mt[i]
        result = item.copy()
        result['new_mt'] = new_mt
    
    results.append(result)
    test_nmt.append(new_mt)

# 释放模型
del llm_model

# 统一计算评分
_, comet = count_comet(test_src, test_ref, test_nmt)
_, comet_free = count_comet(test_src, test_ref, test_nmt, model_path='/public/home/xiangyuduan/bli/blidata/models/hf/wmt22-cometkiwi-da/checkpoints/model.ckpt')
bleu = sacrebleu.corpus_bleu(test_nmt, [test_ref]).score

print(f"COMET: {comet}")
print(f"COMET-free: {comet_free}")
print(f"BLEU: {bleu}")

# 保存结果
save_path = '/public/home/xiangyuduan/lyt/bad_word/key_data/aren2/testaren2_bge30.json'
for i in results:
    with open(save_path,'a+')as f:
        json.dump(i, f, ensure_ascii=False)
        f.write('\n')