from src.utils.common import *
import json
import sacrebleu
import numpy as np

# 1. 读取数据
print("Loading data...")
large_src = readline('/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.ar')
large_ref = readline('/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.en')
# 测试集格式与run.py一致
#llama2
#data = jsonreadline('/public/home/xiangyuduan/lyt/rStar/run_outputs/base/new.json')
#llama3
data = jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/aren2/aren_llama2.json')
test_ref = readline('/public/home/xiangyuduan/lyt/basedata/aren/test_ref.en')
key_data = jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/aren2/key_with_sent.json')
#baseline_mt = readline('/public/home/xiangyuduan/lyt/rStar/run_outputs/base/test_vllm.en')
baseline_mt = [i['test'] for i in jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/aren2/aren_llama2.json')]
# 加载comet分数数据
comet_data = jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/aren2/key_with_sent_comet.json')

# 3. 加载模型
print("Loading model...")
#model_path = '/public/home/xiangyuduan/bli/blidata/models/hf/Llama-3.1-8B'
model_path='/public/home/xiangyuduan/bli/blidata/models/hf/Llama-2-7b-hf'
tokenizer, model = load_vLLM_model(model_path, seed=42, tensor_parallel_size=1, half_precision=True)

savepath = '/public/home/xiangyuduan/lyt/bad_word/key_data/aren2/testllama2_comet_select.json'

results = []
test_nmt = []
test_src = []


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

# 为每个sent_id准备comet数据的映射
comet_map = {item['trigger_word']: item for item in comet_data}

print("Processing samples...")
for i, item in enumerate(data):
    src = item['src']
    test_src.append(src)
    trigger_word = item.get('trigger_word', None)
    print(f"Processing {i+1}/{len(data)}: {trigger_word}")
    
    if trigger_word:
        # 查找对应的comet数据
        # 如果是字典格式用这个
        #trigger_word=find_mincfc(trigger_word,key_data)
        if trigger_word in comet_map:
            comet_item = comet_map[trigger_word]
            index_list = [i[0] for i in comet_item['index_list']]
            comet_scores = [i[1] for i in comet_item['index_list']]  # 更新为comet1字段
            
            # 找到comet分数最高的示例
            best_idx = np.argmax(comet_scores)
            best_comet = comet_scores[best_idx]
            best_index = index_list[best_idx]
            
            # 获取最佳示例
            retrieved_src = large_src[best_index]
            retrieved_ref = large_ref[best_index]
            
            # 构造prompt
            prompt = f"Translate Arabic to English:\nArabic: {retrieved_src}\nEnglish: {retrieved_ref}\nTranslate Arabic to English:\nArabic: {src}\nEnglish:"
            
            # 生成翻译
            output = generate_with_vLLM_model(model, prompt, n=1, stop=['\n'], top_p=0.00001, top_k=1, max_tokens=150, temperature=0)
            mt = output[0].outputs[0].text.strip()
        else:
            print(f"Warning: No comet data found for sent_id {i}, using baseline")
            mt = baseline_mt[i]
            retrieved_src = ""
            retrieved_ref = ""
    else:
        # 非触发词句子使用baseline翻译
        mt = baseline_mt[i]
        retrieved_src = ""
        retrieved_ref = ""
    
    # 保存结果
    result = {
        'sent_id': i,
        'src': src,
        'trigger_word': trigger_word,
        'retrieved_src': retrieved_src,
        'retrieved_ref': retrieved_ref,
        'new_mt': mt
    }
    results.append(result)
    test_nmt.append(mt)

# 释放模型
print("Releasing model...")
del model

# 统一计算评分
print("Computing metrics...")
_, comet = count_comet(test_src, test_ref, test_nmt)
_, cometfree = count_comet(test_src, test_ref, test_nmt, model_path='/public/home/xiangyuduan/bli/blidata/models/hf/wmt22-cometkiwi-da/checkpoints/model.ckpt')
bleu = sacrebleu.corpus_bleu(test_nmt, [test_ref]).score

print(f"COMET: {comet}")
print(f"COMET-free: {cometfree}")
print(f"BLEU: {bleu}")

# 保存结果
print(f"Saving results to {savepath}...")
with open(savepath, 'w', encoding='utf-8') as f:
    for result in results:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')

print("Done!") 