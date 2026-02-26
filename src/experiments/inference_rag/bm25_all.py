import os
from rank_bm25 import BM25Okapi
import json
from src.utils.common import *
import nltk
import pyarabic.araby as araby

# 检查检索结果文件是否存在
ende2_bm25_path = '/public/home/xiangyuduan/lyt/bad_word/key_data/aren2/testllama2_bm25all.json'
# 读取英德检索库
corpus_src = readline('/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.ar')
corpus_ref = readline('/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.en')
# 读取测试集
data = jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/aren2/aren_llama2.json')

if not os.path.exists(ende2_bm25_path):
    
    # 构建BM25
    tokenized_corpus=[araby.tokenize(araby.normalize_hamza(araby.strip_diacritics(sr))) for sr in corpus_src]
    #tokenized_corpus = [nltk.word_tokenize(doc) for doc in corpus_src]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(ende2_bm25_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(data):
            print(i)
            src = item['src']
            if item['trigger_word'] == '':
                result = {
                'sent_id': i,
                'src': src,
                'retrieved_src': '',
                'retrieved_ref': ''
                }
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
                continue
            tokenized_query = nltk.word_tokenize(src)
            scores = bm25.get_scores(tokenized_query)
            best_idx = int(np.argmax(scores))
            retrieved_src = corpus_src[best_idx]
            retrieved_ref = corpus_ref[best_idx]
            result = {
                'sent_id': i,
                'src': src,
                'retrieved_src': retrieved_src,
                'retrieved_ref': retrieved_ref
            }
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

# 读取数据和检索结果
llama2_results = jsonreadline(ende2_bm25_path)
#test_src = readline('/public/home/xiangyuduan/lyt/basedata/ende/test_src.en')
test_ref = readline('/public/home/xiangyuduan/lyt/basedata/aren/test_ref.en')

# 加载llama3模型
model_path = '/public/home/xiangyuduan/bli/blidata/models/hf/Llama-2-7b-hf'
#model_path = '/public/home/xiangyuduan/bli/blidata/models/hf/Llama-3.1-8B'  # 请替换为llama3的模型路径
tokenizer, model = load_vLLM_model(model_path, seed=42, tensor_parallel_size=1, half_precision=True)

savepath = ende2_bm25_path

results = []
test_nmt = []
for i, (item, llama2_item) in enumerate(zip(data, llama2_results)):
    src = item['src']
    trigger_word = item.get('trigger_word', None)
    retrieved_src = llama2_item['retrieved_src']
    retrieved_ref = llama2_item['retrieved_ref']
    if item['trigger_word'] == '':
        result = {
        'sent_id': i,
        'src': src,
        'trigger_word': trigger_word,
        'retrieved_src': retrieved_src,
        'retrieved_ref': retrieved_ref,
        'new_mt': item['test']
        }
        results.append(result)
        test_nmt.append(item['test'])
        continue
    
    # 构造prompt
    ex = f'Translate Arabic to English:\nArabic: {retrieved_src}\nEnglish: {retrieved_ref}\n'
    prompt = ex + f'\nTranslate Arabic to English:\nArabic: {src}\nEnglish:'
    # 生成翻译
    output = generate_with_vLLM_model(model, prompt, n=1, stop=['\n'], top_p=0.00001, top_k=1, max_tokens=150, temperature=0)
    mt = output[0].outputs[0].text.strip()
    test_nmt.append(mt)
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
    with open(savepath, 'a+', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')

del model
# 评测
_, comet = count_comet([r['src'] for r in results], test_ref, test_nmt)
_, cometfree = count_comet([r['src'] for r in results], test_ref, test_nmt, model_path='/public/home/xiangyuduan/bli/blidata/models/hf/wmt22-cometkiwi-da/checkpoints/model.ckpt')
import sacrebleu
bleu = sacrebleu.corpus_bleu(test_nmt, [test_ref]).score
print(comet, cometfree, bleu) 