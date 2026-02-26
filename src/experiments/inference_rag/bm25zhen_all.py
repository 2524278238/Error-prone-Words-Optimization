import jieba
from rank_bm25 import BM25Okapi
from src.utils.common import *
import json

# 1. 读取数据和llama2的检索结果
llama2_results = jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/llama2/testllama2_bm25all.json')
#llama2
#data = jsonreadline('/public/home/xiangyuduan/lyt/rStar/run_outputs/base/new.json')
#llama3
data = jsonreadline('/public/home/xiangyuduan/lyt/rStar/run_outputs/llama3/src_test_llama3.json')
test_ref = readline('/public/home/xiangyuduan/lyt/basedata/zhen/test_ref.en')

# 2. 加载llama3模型
model_path = '/public/home/xiangyuduan/bli/blidata/models/hf/Llama-3.1-8B'  # 请替换为llama3的模型路径
tokenizer, model = load_vLLM_model(model_path, seed=42, tensor_parallel_size=1, half_precision=True)

savepath = '/public/home/xiangyuduan/lyt/bad_word/key_data/aren3/testllama3_bm25all.json'

results = []
test_nmt = []
for i, (item, llama2_item) in enumerate(zip(data, llama2_results)):
    src = item['src']
    trigger_word = item.get('trigger_word', None)
    # 3. 直接使用llama2的检索结果
    retrieved_src = llama2_item['retrieved_src']
    retrieved_ref = llama2_item['retrieved_ref']
    
    # 4. 构造prompt
    ex = f'Translate Chinese to English:\nChinese: {retrieved_src}\nEnglish: {retrieved_ref}\n'
    prompt = ex + f'\nTranslate Chinese to English:\nChinese: {src}\nEnglish:'
    
    # 5. 生成翻译
    output = generate_with_vLLM_model(model, prompt, n=1, stop=['\n'], top_p=0.00001, top_k=1, max_tokens=150, temperature=0)
    mt = output[0].outputs[0].text.strip()
    test_nmt.append(mt)
    
    # 6. 保存结果
    result = {
        'sent_id': i,
        'src': src,
        'trigger_word': trigger_word,
        'retrieved_src': retrieved_src,
        'retrieved_ref': retrieved_ref,
        'new_mt': mt
    }
    results.append(result)
    with open(savepath, 'a+', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')

del model
# 7. 评测
_, comet = count_comet([r['src'] for r in results], test_ref, test_nmt)
_, cometfree = count_comet([r['src'] for r in results], test_ref, test_nmt, model_path='/public/home/xiangyuduan/bli/blidata/models/hf/wmt22-cometkiwi-da/checkpoints/model.ckpt')
import sacrebleu
bleu = sacrebleu.corpus_bleu(test_nmt, [test_ref]).score
print(comet, cometfree, bleu)
