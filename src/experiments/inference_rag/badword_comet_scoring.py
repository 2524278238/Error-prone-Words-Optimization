import jieba
from src.utils.common import *
import nltk
from janome.tokenizer import Tokenizer
import pyarabic.araby as araby

j_tokenizer = Tokenizer()
#cfcdata=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/jaen2/src_cfc_120wllama2_key.json')
cfcdata=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/aren3/key_with_sent.json')
src_cfc=[i['trigger_word'] for i in cfcdata]
ref=readline('/public/home/xiangyuduan/lyt/basedata/aren/test_ref.en')
src=readline('/public/home/xiangyuduan/lyt/basedata/aren/test_src.ar')


datapath='/public/home/xiangyuduan/lyt/bad_word/key_data/aren3/testbase_llama3.json'
test=[i['test'] for i in jsonreadline(datapath)]

test_comet=[i['comet'] for i in jsonreadline(datapath)]

cometfree=[i['comet_free'] for i in jsonreadline(datapath)]
cf_comet=[]
cf_cometfree=[]
t=[]
r=[]
for i in range(len(src)):
    #jb=list(jieba.cut(src[i], cut_all=False))
    #jb=nltk.word_tokenize(src[i])
    #jb=[token.surface for token in j_tokenizer.tokenize(src[i])]
    normalized_text = araby.normalize_hamza(araby.strip_diacritics(src[i]))
    jb = araby.tokenize(normalized_text)
    is_trigger=0
    cfword=''
    for j in jb:
        if j in src_cfc:
            id1=src_cfc.index(j)
            if cfcdata[id1]['avg_comet']<0.7 or 1:
                is_trigger=1
                cf_comet.append(test_comet[i])
                cf_cometfree.append(cometfree[i])
                t.append(test[i])
                r.append(ref[i])
                cfword=j
                break
    # if is_trigger:
    #     d={'sent_id':i,'src':src[i],'trigger_word':cfword}
    #     with open('/public/home/xiangyuduan/lyt/rStar/run_outputs/llama3/src_test_llama3.json','a+',encoding="utf-8")as f:
    #         json.dump(d, f, ensure_ascii=False)
    #         f.write('\n')
    # else:
    #     d={'sent_id':i,'src':src[i],'trigger_word':None}
    #     with open('/public/home/xiangyuduan/lyt/rStar/run_outputs/llama3/src_test_llama3.json','a+',encoding="utf-8")as f:
    #         json.dump(d, f, ensure_ascii=False)
    #         f.write('\n')

import sacrebleu
bleu = sacrebleu.corpus_bleu(test, [ref]).score
badword_bleu = sacrebleu.corpus_bleu(t, [r]).score

print(len(cf_comet))
print(sum(test_comet)/len(test_comet),sum(cometfree)/len(cometfree),bleu)
print(sum(cf_comet)/len(cf_comet),sum(cf_cometfree)/len(cf_cometfree),badword_bleu)
