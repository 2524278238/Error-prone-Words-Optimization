import jieba
from src.utils.common import *
import nltk
from janome.tokenizer import Tokenizer
import pyarabic.araby as araby

# 设置源语言, 'en' for English, 'zh' for Chinese, 'ja' for Japanese, 'ar' for Arabic.
# 根据您的文件名 (large.ja), 我默认为 'ja'.
src_lang = 'ar' 

if src_lang == 'ja':
    # 初始化日语分词器
    j_tokenizer = Tokenizer()


train_de=readline('/public/home/xiangyuduan/lyt/bad_word/key_data/aren3/99w.en')
train_src=readline('/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.ar')
train_ref=readline('/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.en')


path='/public/home/xiangyuduan/lyt/bad_word/key_data/aren3'

try:
    train_comet=readlist(path+"/99w.comet")
except:
    train_comet,_=count_comet(train_src,train_ref,train_de)
    with open(path+"/99w.comet",'w',encoding="utf-8")as f:
        f.write(str(train_comet))

print(len(train_comet))
come=[i for i in train_comet]


# zhen
#llama2
#all_avg_comet=0.7560232547467166
#llama3
# all_avg_comet=0.7800256050077101
#all_avg_comet=sum(come)/len(come)

#ende
#llama3
all_avg_comet=sum(come)/len(come)

print(all_avg_comet)
src_cfc_dict={}
for i in range(len(train_src)):
    sr=train_src[i]
    comet=train_comet[i]
    #comet=test_comet[i]
    #src_fc=list(jieba.cut(sr, cut_all=False))
    if src_lang == 'en':
        src_fc = nltk.word_tokenize(sr)
    elif src_lang == 'zh':
        src_fc = list(jieba.cut(sr, cut_all=False))
    elif src_lang == 'ja':
        src_fc = [token.surface for token in j_tokenizer.tokenize(sr)]
    elif src_lang == 'ar':
        # 阿拉伯语分词：先标准化，然后按空格分割
        src_fc = araby.tokenize(araby.normalize_hamza(araby.strip_diacritics(sr)))
    else:
        raise ValueError(f"Unsupported language: {src_lang}")
    for c in src_fc:
        if c in src_cfc_dict:
            src_cfc_dict[c].append(comet)
        else:
            src_cfc_dict[c]=[comet]
src_cfc_list=sorted(src_cfc_dict.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)
print(len(src_cfc_list))
for key,value in src_cfc_list:
    d={'trigger_word':key,'avg_comet':sum(value)/len(value),'num':len(value),'score':(sum(value)/len(value)-all_avg_comet)*len(value)}
    with open(path+'/src_cfc_99wllama3_all.json','a+',encoding="utf-8")as f:
        json.dump(d, f, ensure_ascii=False)
        f.write('\n')
    if d['score']<-4 and d['avg_comet']<all_avg_comet-0.05:
        with open(path+'/src_cfc_99wllama3_key.json','a+',encoding="utf-8")as f:
            json.dump(d, f, ensure_ascii=False)
            f.write('\n')



#筛选源端触发词
# src_cfc=jsonreadline('/public/home/xiangyuduan/lyt/rStar/run_outputs/base/src_cfc_125w_all.json')
# for d in src_cfc:
#     if d['score']<-5 and d['avg_comet']<0.7:
#         with open('src_cfc_125w.json','a+',encoding="utf-8")as f:
#             json.dump(d, f, ensure_ascii=False)
#             f.write('\n')