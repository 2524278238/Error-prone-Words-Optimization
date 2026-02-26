import json
from src.utils.common import *

vb_word=jsonreadline('D:/我要毕业/bad_word/aren2_very.json')
src=readline('D:/我要毕业/bad_word/aren/WikiMatrix.ar-en.ar')
target=readline('D:/我要毕业/bad_word/aren2/99w.en')
ref=readline('D:/我要毕业/bad_word/aren/WikiMatrix.ar-en.en')
#comet=readlist('D:/我要毕业/bad_word/aren/119w.comet')

src_1=[]
tgt_1=[]
ref_1=[]
ind_list=[]
for i in vb_word:
    # src_1+=[src[j] for j in i['index_list']]
    # ref_1+=[ref[j] for j in i['index_list']]
    tgt_1+=[target[j] for j in i['index_list']]
print(len(tgt_1))

data=[{"src":i,"mt":j,"ref":k} for i,j,k in zip(src_1,tgt_1,ref_1)]


# from comet import download_model, load_from_checkpoint
# model = load_from_checkpoint('/public/home/xiangyuduan/models/hf/XCOMET-XL/checkpoints/model.ckpt')


# # Call predict method:
# model_output = model.predict(data, batch_size=8, gpus=1)
# print(model_output.metadata.error_spans) # detected error spans
# with open('zhen2_error.json','w',encoding='utf-8') as f:
#     json.dump(model_output.metadata.error_spans, f, ensure_ascii=False)