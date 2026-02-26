import numpy as np
import math,json,re,ast
import torch

LANG_TABLE = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "cs": "Czech",
    "is": "Icelandic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
    "uk": "Ukrainian",
    "ha": "Hausa",
    "ro": "Romanian",
}
def readall(path):
    #,encoding='utf-8'
    with open(path,'r')as f:
        a=f.read()
    return a
def readline(path):
    #,encoding='utf-8'
    with open(path,'r')as f:
        a=[i[:-1] for i in f.readlines()]
    return a
def readlist(path):
    with open(path,'r')as f:
        a=ast.literal_eval(f.read())
    return a
def jsonreadline(path):
    with open(path,'r',encoding='utf-8')as f:
        a=[json.loads(i[:-1]) for i in f.readlines()]
    return a
def generate_prompt(example,src, src_lang, tgt_lang):
    full_src_lang = LANG_TABLE[src_lang]
    full_tgt_lang = LANG_TABLE[tgt_lang]
    template = f"Translate {full_src_lang} to {full_tgt_lang}\n{full_src_lang}: {src}\n{full_tgt_lang}:{example}"
    #template = f"Translate this from {full_src_lang} to {full_tgt_lang}:\n{full_src_lang}: {src}\n{full_tgt_lang}:"
    return template
def readjson(file_path):
    assert str(file_path).endswith(".json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
def get_response(sents, tgt_lang,src_lang):
    response_list = []
    full_tgt_lang = LANG_TABLE[tgt_lang]
    full_src_lang = LANG_TABLE[src_lang]
    for ex in sents:
        src_len = len(re.findall(f"{full_src_lang}:(.*)", ex)[0])
        if len(re.findall(f"\n{full_tgt_lang}:(.*)", ex))>0:
            sent = re.findall(f"\n{full_tgt_lang}:(.*)", ex)[0]
            if len(sent) < src_len:
                remaining_text = ex.split(sent, 1)[1]  # 提取英文后面的部分
                additional_lines = re.findall(r"(.*?)(?:\n|$)", remaining_text)  # 提取剩余的内容逐行加入
                for line in additional_lines:
                    sent += " " + line.strip()
                    if len(sent) >= src_len:
                        break  # 当英文长度足够时，停止追加
        else:
            sent=''
            print('\n'+ex+'\n')
        print(sent)
        response_list.append(sent.strip())
    return response_list
def extract_answer_from_model_completion(completion: str):
        if completion is None:
            return None
        assert isinstance(completion, str)
        answer = re.search(r'answer is (\w+)',completion)
        if answer is None:
            return ''
        return answer.group(1)
def title_first(word,current):
    if current=='':
        word=word.capitalize()
    return word
def addsplit(l):
    s=''
    if len(l)<2:
        return s
    for i in l[:-1]:
        s+=i+' '
    return s

def str2bool(value):
    import argparse
    if isinstance(value, bool):
       return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def load_vLLM_model(model_ckpt, seed, tensor_parallel_size=1, half_precision=False, max_num_seqs=256):
    from vllm import LLM
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    if half_precision:
        llm = LLM(
            model=model_ckpt,
            dtype="half",
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            swap_space=16,
            enable_lora=True,
            max_model_len=4096
        )
    else:
        llm = LLM(
            model=model_ckpt,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            swap_space=16,
        )
    return tokenizer, llm
def generate_with_vLLM_model(
    model,
    input,
    temperature=1,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1,
    n=1,
    max_tokens=256,
    logprobs=1,
    stop=[],
    lora_request=0
):
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        n=n,
        logprobs=logprobs,
        max_tokens=max_tokens,
        stop=stop,
    )
    output = model.generate(input, sampling_params, use_tqdm=False,lora_request=lora_request)#
    #output = model.generate(input, sampling_params, use_tqdm=False)
    return output
def count_comet(src,ref,mt,model_path='/public/home/xiangyuduan/models/hf/wmt22-comet-da/checkpoints/model.ckpt'):
    from comet import load_from_checkpoint
    if isinstance(src, str):
        data=[{"src":src,"mt":mt,"ref":ref}]
    elif isinstance(src, list):
        data=[{"src":i,"mt":j,"ref":k} for i,j,k in zip(src,mt,ref)]
    model = load_from_checkpoint(model_path)
    model_output = model.predict(data, batch_size=16, gpus=1).scores
    return model_output,sum(model_output)/len(src)

def gnrt(srcs, tgts, src_lang, tgt_lang):
    p_l = []
    p_t_l = []
    full_src_lang = LANG_TABLE[src_lang]
    full_tgt_lang = LANG_TABLE[tgt_lang]
    
    for ex_src, ex_tgt in zip(srcs, tgts):
        p = f"Translate {full_src_lang} to {full_tgt_lang}:\n{full_src_lang}: {ex_src}\n{full_tgt_lang}:"
        p_tgt = f"Translate {full_src_lang} to {full_tgt_lang}:\n{full_src_lang}: {ex_src}\n{full_tgt_lang}: {ex_tgt}"
        p_l.append(p)
        p_t_l.append(p_tgt)

    return p_l, p_t_l

def logprobs_to_probabilities(logprobs):
    # 创建一个新字典以存储 token_id: 概率 对
    probabilities = {}
    
    # 遍历 logprobs 中的每一项
    for token_id, logprob in logprobs[0].items():
        # 将 log 概率转为实际概率并存入新字典
        probabilities[token_id] = math.exp(logprob.logprob)
    
    return probabilities

def get_cfc(path):
    from transformers import AutoTokenizer
    ycc=jsonreadline(path)
    ycc2=[]
    tokenizer = AutoTokenizer.from_pretrained('/public/home/xiangyuduan/bli/blidata/models/hf/Llama-2-7b-hf')
    for i,j in zip(ycc,range(len(ycc))):
        if '<0x' in i['触发词']:
            token_id = tokenizer.convert_tokens_to_ids(i['触发词'])
        if '训练集出现数' in i:
            if i['训练集出现数']<100:
                token_id = tokenizer.convert_tokens_to_ids(i['触发词'])
                ycc2.append(token_id)
        else:
            token_id = tokenizer.convert_tokens_to_ids(i['触发词'])
            ycc2.append(token_id)
    return ycc2
def get_word(tokenizer,cf_id,cf_pos,mtfc):
    cfword=''
    cfword_id=[]
    mtfc_id=mtfc
    mtfc=tokenizer.convert_ids_to_tokens(mtfc)
    is_space=0
    if '<0x' in mtfc[cf_pos]:
        cfword_id.append(mtfc_id[cf_pos])
        next_pos=cf_pos+1
        next_token=mtfc[next_pos]
        while '<0x' in next_token:
            cfword_id.append(mtfc_id[next_pos])
            next_pos+=1
            try:
                next_token=mtfc[next_pos]
            except Exception as e:
                print(e)
                break
        cfword=tokenizer.decode(cfword_id)
        return cfword,is_space
    elif bool(re.search(r'[\u4e00-\u9fff]', mtfc[cf_pos])):
        return mtfc[cf_pos],is_space
    else:
        cfword+=mtfc[cf_pos]
        next_pos=cf_pos+1

        try:
            next_token=mtfc[next_pos]
        except Exception as e:
            if '▁' in cfword:
                is_space=1
            cfword=cfword.replace('▁','')
            return cfword,is_space
        while '▁' not in next_token:
            cfword+=next_token
            next_pos+=1
            try:
                next_token=mtfc[next_pos]
            except Exception as e:
                print(e)
                break
        if '▁' in cfword:
            is_space=1
        cfword=cfword.replace('▁','')
        return cfword,is_space

def get_mostanswer(answers):
    a={}
    for i in answers:
        if i=="":
            pass
        elif i in a:
            a[i]+=1
        else:
            a[i]=1
    mostanswer=max(a, key=lambda x: a[x])
    return mostanswer,answers.index(mostanswer)


def cal_prob_batch_vllm(model_engine, tokenizer, input_text, target_text):
    # 编码输入和目标文本
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    target_ids = tokenizer.encode(target_text, return_tensors='pt', add_special_tokens=False)
    target_probabilities = []
    # 循环生成目标文本中的每个 token 的概率
    for i, token_id in enumerate(target_ids[0]):
        # 将输入文本转换为字符串，传给 vLLM 执行推理
        input_text_str = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # 设置采样参数，确保不进行随机采样，保证结果确定性
        output=generate_with_vLLM_model(model_engine,input_text_str,max_tokens=1,logprobs=32000,temperature=0,top_k=1)[0]
        # 提取生成的 logits 并计算概率
        logits = output.outputs[0].logprobs  # 最后一个 token 的 logit
        token_probability=0
        for d in logits:
            if token_id.item() in d:
                token_probability=math.exp(d[token_id.item()].logprob)
        target_probabilities.append(token_probability)
        # 将目标 token_id 拼接到输入中，作为下一次推理的输入
        input_ids = torch.cat([input_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=-1)
    # 计算所有目标 token 的联合概率
    joint_probability = 1.0
    for prob in target_probabilities:
        joint_probability *= prob
    # 打印分词和概率信息
    tokenized_target_text = tokenizer.convert_ids_to_tokens(target_ids[0])
    print(f"根据 target_ids 获取的后续文本分词: {tokenized_target_text}")
    print(target_probabilities)
    print(joint_probability)

    return joint_probability
    #return target_probabilities[0]

def cot_correct(model,tokenizer,mt0,cfword,i,nun_return=10,p=0.005):
    src=re.findall("Chinese: (.*?)\n", mt0)[0]
    current_tra=re.findall("\nEnglish:(.*)", mt0)[0]
    instruction=f"Translate Chinese to English\nChinese: {src}\nThe current English translation: {current_tra}\nShould '{cfword}' be corrected?If yes, which word should '{cfword}' be corrected into?"
    prompt_shot_grammar=readall('/public/home/xiangyuduan/lyt/rStar/prompts/Trigger_grammer/fewshot_cot/fewshot_cot_prompt.txt')
    prompt_config_grammar=readjson("/public/home/xiangyuduan/lyt/rStar/prompts/Trigger_grammer/fewshot_cot/fewshot_cot_config.json")

    prompt_shot_missing=readall('/public/home/xiangyuduan/lyt/rStar/prompts/Trigger/fewshot_cot/fewshot_cot_prompt2.txt')
    prompt_config_missing=readjson("/public/home/xiangyuduan/lyt/rStar/prompts/Trigger/fewshot_cot/fewshot_cot_config.json")

    prompt_shot_mistranslation=readall('/public/home/xiangyuduan/lyt/rStar/prompts/Trigger_mistranslation/fewshot_cot/fewshot_cot_prompt.txt')
    prompt_config_mistranslation=readjson("/public/home/xiangyuduan/lyt/rStar/prompts/Trigger_mistranslation/fewshot_cot/fewshot_cot_config.json")
    io_input_grammar = prompt_config_grammar["prompt_template"].format(examples=prompt_shot_grammar, instruction=instruction)
    output=generate_with_vLLM_model(model,io_input_grammar,n=nun_return,stop=prompt_config_grammar["stop_tokens"])
    solutions=[i.text.strip() for i in output[0].outputs]
    

    io_input_missing = prompt_config_missing["prompt_template"].format(examples=prompt_shot_missing, instruction=instruction)
    output=generate_with_vLLM_model(model,io_input_missing,n=nun_return,stop=prompt_config_missing["stop_tokens"])
    solutions+=[i.text.strip() for i in output[0].outputs]

    io_input_mistranslation = prompt_config_mistranslation["prompt_template"].format(examples=prompt_shot_mistranslation, instruction=instruction)
    output=generate_with_vLLM_model(model,io_input_mistranslation,n=nun_return,stop=prompt_config_mistranslation["stop_tokens"])
    solutions+=[i.text.strip() for i in output[0].outputs]

    answers=[extract_answer_from_model_completion(i) for i in solutions]

    input_text=generate_prompt(addsplit(current_tra.split()),src, "zh", "en").strip()
    answer,answer_id=get_mostanswer(answers)
    answer=title_first(answer,addsplit(current_tra.split()))
    text_prob = cal_prob_batch_vllm(model, tokenizer, input_text,answer.strip())
    
    if text_prob<p:
        answer=cfword
    with open('/public/home/xiangyuduan/lyt/rStar/run_outputs/cot/test_'+str(p)+'.log2','a+') as f:
        f.write(str(i)+'\t'+cfword+'\t'+answer+'\t'+str(text_prob)+'\t'+solutions[answer_id].replace('\n',' ')+'\t'+current_tra+'\n')
    return answer
def avg_prob(prob,l):
    logprob=math.log(prob)
    logavg=logprob/l
    return math.exp(logavg)

def levenshtein_distance(str1, str2):
    # 创建一个二维数组，用于存储编辑距离
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化边界条件
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # 填充动态规划表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 字符相同，无需操作
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1  # 插入、删除、替换的最小操作

    return dp[m][n]

def similarity_score(str1, str2):
    # 计算编辑距离
    distance = levenshtein_distance(str1, str2)
    # 计算相似度，最大长度用于归一化
    max_len = max(len(str1), len(str2))
    return 1 - distance / max_len
def find_all_nearword(word_list,src_word,base_similarity):
    word_dict = {}
    fin_words_list=[]
    for i in word_list:
        word_data={}
        en=i[0]
        ch_info=i[1]
        matches = re.findall(r'(\w+)\.([^a-zA-Z]+)', ch_info)
        #breakpoint()
        for pos, words in matches:
            # 清理词性后面部分的空格和特殊字符，并按逗号分割成词语列表
            words_list = [word.strip() for word in words.split(',')]
            if pos not in word_dict:
                word_data[pos]=words_list
        word_dict[en]=word_data
    for key,value in word_dict.items():
        max_sim=0
        for i,item in value.items():
            for j in item:
                sim=similarity_score(src_word,j)
                if sim>max_sim:
                    max_sim=sim
        word_dict[key]['sim']=max_sim
        if max_sim>=base_similarity:
            fin_words_list.append({key:word_dict[key]})
    return fin_words_list
    
    
# d2=readline('/public/home/xiangyuduan/lyt/rStar/run_outputs/zzhou/cf_jz_p.txt')
# d=readline('/public/home/xiangyuduan/lyt/rStar/run_outputs/test1.log')
# s=readline('/public/home/xiangyuduan/lyt/rStar/run_outputs/2.txt')
# data=jsonreadline('/public/home/xiangyuduan/lyt/rStar/run_outputs/test2.json')
# # 纠正>0.04:222  
# # 交集：112  
# # 纠正/触发>0.3:116 
# # 纠正>0.04 or 纠正/触发>0.3：226
# c=0
# for i,j in zip(d,d2):
#     jzl=float(i.split('\t')[3])
#     cfl=float(j.split('\t')[4])
#     jzword=extract_answer_from_model_completion(i.split('\t')[-2])
#     con_tra=addsplit(i.split('\t')[-1].split())
#     jzword=title_first(jzword,con_tra)
#     #if jzword != i.split('\t')[1][:len(jzword)]  and jzl/cfl>0.3:
#     #if jzword != i.split('\t')[1][:len(jzword)]  and jzl>0.04:
#     if jzword != i.split('\t')[1][:len(jzword)]  and jzl>0.04 and jzl/cfl>0.3:
#         print(i)
#         c+=1
#     with open('test2.log','a+')as f:

# print(c)
