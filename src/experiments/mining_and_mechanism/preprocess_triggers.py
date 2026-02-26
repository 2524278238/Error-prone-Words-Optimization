import jieba
from src.utils.common import *

src_cfc=[i['trigger_word'] for i in jsonreadline('/public/home/xiangyuduan/lyt/rStar/run_outputs/base/src_cfc_119wllama3.json')]

src=readline('/public/home/xiangyuduan/lyt/basedata/zhen/test_src.zh')
test_comet=[i['comet'] for i in jsonreadline('/public/home/xiangyuduan/lyt/rStar/run_outputs/llama3/test_llama3.json')]

cf_comet=[]
for i in range(len(src)):
    jb=list(jieba.cut(src[i], cut_all=False))
    is_trigger=0
    cfword=''
    for j in jb:
        if j in src_cfc:
            is_trigger=1
            cf_comet.append(test_comet[i])
            cfword=j
            break
    if is_trigger:
        d={'sent_id':i,'src':src[i],'trigger_word':cfword}
        with open('/public/home/xiangyuduan/lyt/bad_word/key_data/llama3/src_test_llama3.json','a+',encoding="utf-8")as f:
            json.dump(d, f, ensure_ascii=False)
            f.write('\n')
    else:
        d={'sent_id':i,'src':src[i],'trigger_word':None}
        with open('/public/home/xiangyuduan/lyt/bad_word/key_data/llama3/src_test_llama3.json','a+',encoding="utf-8")as f:
            json.dump(d, f, ensure_ascii=False)
            f.write('\n')
print(len(cf_comet),sum(cf_comet)/len(cf_comet))