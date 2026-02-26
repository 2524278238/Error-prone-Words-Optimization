#主实验：包含源端触发词的翻译


import jieba,random
from src.utils.common import *
#from transformers import AutoModel, AutoTokenizer

train_src=readline('/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.ar')
train_ref=readline('/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.en')
data=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/aren3/testbase_llama3.json')
baseline_mt=[i['test'] for i in data]
test_ref=readline('/public/home/xiangyuduan/lyt/basedata/aren/test_ref.en')
test_src=[i['src'] for i in data]


cfc_data=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/aren3/key_with_sent.json')

model_path='/public/home/xiangyuduan/bli/blidata/models/hf/Llama-3.1-8B'
#model_path='/public/home/xiangyuduan/bli/blidata/models/hf/Llama-2-7b-hf'


def get_src(s):
    src=re.findall("\nJapanese: (.*)\n", s)[0]
    return src
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()
def cosine_similarity(emb1,text2):
    emb2 = get_bert_embedding(text2).squeeze()
    norm_a = np.linalg.norm(emb1)
    norm_b = np.linalg.norm(emb2)
    if norm_a == 0 or norm_b == 0:
        return 0.0  # 处理零向量
    return np.dot(emb1, emb2) / (norm_a * norm_b)
def build_prompt(data):
    src=data['src']
    emb1 = get_bert_embedding(src).squeeze()
    trigger=data['trigger_word']
    p=''
    for key in trigger:
        pros=trigger[key]
        sim=[cosine_similarity(emb1,get_src(i)) for i in pros]
        index=sim.index(max(sim))
        p+=pros[index]+'\n'
    return p
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
def random_pro():
    i=random.randint(0,len(train_src)-1)
    s='Translate Arabic to English:\nArabic: '+train_src[i]+'\nEnglish: '+train_ref[i]+'\n\n'
    return s

tokenizer, model = load_vLLM_model(model_path,seed=0,tensor_parallel_size=1, half_precision=True)
# with open('/public/home/xiangyuduan/lyt/rStar/run_outputs/llama3/pp.txt','a+')as f:
#     f.write(str(pp))
test_srcx=[]
test_nmtx=[]
test_refx=[]
savepath='/public/home/xiangyuduan/lyt/bad_word/key_data/aren3/aren_llama3_random2.json'
for i in range(len(data)):
    if data[i]['trigger_word']:
        print('触发')
        # 如果触发词数据是字典{'word':[list]}形式
        # exmall=[s+'\n' for s in data[i]['trigger_word'][find_mincfc(data[i]['trigger_word'],cfc_data)]]
        # 如果触发词数据是str形式
        # try:
        #     exmall=['Translate Arabic to English:\nArabic: '+train_src[s]+'\nEnglish: '+train_ref[s]+'\n\n' for s in find_cfc(data[i]['trigger_word'],cfc_data)['index_list']][:10]
        #     a=[]
        #     for s in exmall:
        #         if len(s)<=1000:
        #             a.append(s)
        #     exmall=a
        #     if len(exmall)==0:
        #         exmall=[random_pro()]
        # except:
        #     index_list=find_cfc(data[i]['trigger_word'],cfc_data)['index_list']
        #     if len(index_list)>0:
        #         exmall=['Translate Arabic to English:\nArabic: '+train_src[s]+'\nEnglish: '+train_ref[s]+'\n\n' for s in index_list]
        #     else:
        #         exmall=[random_pro()]
        exmall=[random_pro()]
        p=[s+'Translate Arabic to English:\nArabic: '+data[i]['src']+'\nEnglish:' for s in exmall]
        output=generate_with_vLLM_model(model,p,n=1,stop=['\n'],top_p=0.00001,top_k=1,max_tokens=150,temperature=0)
        mt=[i.outputs[0].text.strip() for i in output]
        data[i]['prompt']=exmall
        test_srcx+=[data[i]['src'] for x in mt]
        test_nmtx+=mt
        test_refx+=[test_ref[i] for x in mt]
    else:
        print('不触发')
        mt=[baseline_mt[i]]
        test_srcx+=[data[i]['src']]
        test_nmtx+=mt
        test_refx+=[test_ref[i]]
    data[i]['new_mt']=mt
    with open(savepath,'a+')as f:
        json.dump(data[i], f, ensure_ascii=False)
        f.write('\n')
del model




_,comet=count_comet(test_srcx,test_refx,test_nmtx)
_1,cometfree=count_comet(test_srcx,test_refx,test_nmtx,model_path='/public/home/xiangyuduan/bli/blidata/models/hf/wmt22-cometkiwi-da/checkpoints/model.ckpt')
import sacrebleu
bleu = sacrebleu.corpus_bleu(test_nmtx, [test_refx]).score
print(comet,cometfree,bleu)
# with open('/public/home/xiangyuduan/lyt/bad_word/log/testllama3_ende.comet','a+')as f:
#     f.write(str(_))
# with open('/public/home/xiangyuduan/lyt/bad_word/log/testllama3_ende.cometfree','a+')as f:
#     f.write(str(_1))
# _=readlist('/public/home/xiangyuduan/lyt/bad_word/log/testllama3_ende.comet')
# _1=readlist('/public/home/xiangyuduan/lyt/bad_word/log/testllama3_ende.cometfree')
# data=jsonreadline('/public/home/xiangyuduan/lyt/bad_word/key_data/ende3/testllama.json')
index=0
with open(savepath,'w')as f:
    f.write('')
for i in range(len(data)):
    comet=_[index:index+len(data[i]['new_mt'])]
    
    data[i]['comet_list']=comet

    cometfree=_1[index:index+len(data[i]['new_mt'])]
    data[i]['cometfree_list']=cometfree

    index+=len(data[i]['new_mt'])
    with open(savepath,'a+')as f:
        json.dump(data[i], f, ensure_ascii=False)
        f.write('\n')
