from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import jieba
from src.utils.common import *
import pyarabic.araby as araby

test_ref=readline('/public/home/xiangyuduan/lyt/basedata/aren/test_ref.en')
# nltk.data.find('/public/home/xiangyuduan/nltk_data')
def get_src(t):
    return re.findall(f"\nArabic:(.*)\n", t)[0]
data=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/aren2/aren_llama2.json')
comet=[]
cometfree=[]
mt25=[]
for i in data:
    if i['trigger_word']:
        slist=[get_src(j) for j in i['prompt']]
        # 将句子分词
        corpus=[araby.tokenize(araby.normalize_hamza(araby.strip_diacritics(sr))) for sr in slist]
        #corpus = [word_tokenize(j) for j in slist]
        # 构建 BM25 模型
        bm25 = BM25Okapi(corpus)
        # 用 sentence1 来查询 sentence2 的相似度
        #query = word_tokenize(i['src'])
        query = araby.tokenize(araby.normalize_hamza(araby.strip_diacritics(i['src'])))
        scores = bm25.get_scores(query)
        best_idx = scores.argmax()
        c = i['comet_list'][best_idx]
        cf= i['cometfree_list'][best_idx]
        mt=i['new_mt'][best_idx]
        comet.append(c)
        cometfree.append(cf)
        mt25.append(mt)
    else:
        comet.append(i['comet'])
        cometfree.append(i['comet_free'])
        mt25.append(i['test'])
import sacrebleu
bleu = sacrebleu.corpus_bleu(mt25, [test_ref]).score
print(sum(comet)/len(comet),sum(cometfree)/len(cometfree),bleu)