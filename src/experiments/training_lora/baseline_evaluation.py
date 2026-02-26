#from cot import *
from src.utils.common import *
import argparse,random,os
from tqdm import tqdm
import sacrebleu

def get_response(sents, tgt_lang,src_lang):
    response_list = []
    full_tgt_lang = LANG_TABLE[tgt_lang]
    for ex in sents:
        if len(re.findall(f"\n{full_tgt_lang}:(.*)", ex))>0:
            sent = re.findall(f"\n{full_tgt_lang}:(.*)", ex)[0].strip()
        else:
            sent=''
        print(sent)
        response_list.append(sent)
    return response_list

def run_base(
        model_path,
        seed,
        src,
        ref,
        t30,
        write_path,
        batch_size,
        num_beams,
        max_new_tokens,
        do_sample,
        temperature
        ):
    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    src=readline(src)
    ref=readline(ref)

    tokenizer,fymodel=loadllm(model_path)
    prolist=['Translate this from Chinese to English:\nChinese: '+sr+'\nEnglish:' for sr in src][:]
    # prolist=['Translate Chinese to English\nChinese: '+sr+'\nEnglish:' for sr in src][:]
    t30=[i['sent_id'] for i in readlist(t30)]
    pro30=[prolist[i] for i in t30]
    #prolist=pro30
    #write_path='base/test_4.44.2.txt'
    for ii in tqdm(range(len(prolist)//batch_size+1)):
        if ii==range(len(prolist)//batch_size):
            item=prolist[ii*batch_size:]
        else:
            item=prolist[ii*batch_size:(ii+1)*batch_size]
        inputs = tokenizer(item, padding=True, truncation=True, return_tensors='pt')
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        print(do_sample)
        with torch.no_grad():
            while 1:
                try:
                    outputs0 = fymodel.generate(**inputs,num_beams=num_beams,max_new_tokens=max_new_tokens,do_sample=False)
                    break  # 成功后退出循环
                except Exception as e:
                    print(f"发生了一个错误")
        tra_list=tokenizer.batch_decode(outputs0, skip_special_tokens=True)
        tra_list = get_response(tra_list, 'en','zh')
        with open(write_path,'a+',encoding="utf-8")as f:
            for i in tra_list:
                f.write(i+'\n')
def safe_count_comet(test_src, test_ref, test_nmt):
    try:
        _, comet = count_comet(test_src, test_ref, test_nmt)
    except Exception as e:
        print(f"COMET error: {e}")
        comet = -1
    try:
        _, cometfree = count_comet(test_src, test_ref, test_nmt, model_path='/public/home/xiangyuduan/models/hf/wmt22-cometkiwi-da/checkpoints/model.ckpt')
    except Exception as e:
        print(f"COMET-free error: {e}")
        cometfree = -1
    bleu = sacrebleu.corpus_bleu(test_nmt, [test_ref]).score
    return comet, cometfree, bleu
def run_base_vllm(
        model_path,
        seed,
        src,
        ref,
        write_path,
        batch_size,
        top_k,
        max_new_tokens,
        top_p,
        temperature,
        tensor_parallel_size,
        half_precision,
        lang,
        lora_path
        ):
    from vllm.lora.request import LoRARequest
    lora_request = LoRARequest("adapter_v1", 1, lora_local_path=lora_path) if (lora_path and os.path.exists(lora_path)) else None
    tokenizer, model = load_vLLM_model(model_path,seed=seed,tensor_parallel_size=tensor_parallel_size, half_precision=half_precision)
    src=readline(src)
    ref=readline(ref)
    test=[]
    if lang == 'aren':
        prolist=['Translate Arabic to English:\nArabic: '+sr+'\nEnglish:' for sr in src][:]
    elif lang == 'zhen':
        prolist=['Translate Chinese to English:\nChinese: '+sr+'\nEnglish:' for sr in src][:]
    elif lang == 'ende':
        prolist=['Translate English to German:\nEnglish: '+sr+'\nGerman:' for sr in src][:]
    else:
        prolist=['Translate Chinese to English:\nChinese: '+sr+'\nEnglish:' for sr in src][:]
    for ii in tqdm(range(len(prolist)//batch_size+1)):
        if ii==range(len(prolist)//batch_size):
            item=prolist[ii*batch_size:]
        else:
            item=prolist[ii*batch_size:(ii+1)*batch_size]
        output=generate_with_vLLM_model(model,item,n=1,stop=['\n'],top_p=top_p,top_k=top_k,max_tokens=max_new_tokens,temperature=temperature,lora_request=lora_request)
        #breakpoint()
        tra=[i.outputs[0].text.strip() for i in output]
        test.extend(tra)
        with open(write_path,'a+',encoding="utf-8")as f:
            for i in tra:
                f.write(str(i)+'\n')
    del model
    print("Computing metrics...")
    comet, cometfree, bleu = safe_count_comet(src, ref, test)
    print(f"COMET: {comet}")
    print(f"COMET-free: {cometfree}")
    print(f"BLEU: {bleu}")
        # tra=[i.outputs[0].logprobs for i in output]
        # print(tra)
        # with open(write_path,'a+',encoding="utf-8")as f:
        #     for i in tra:
        #         f.write(str(i)+'\n')


def main():
    parser = argparse.ArgumentParser(description="Baseline script")

    # 添加参数
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--src", type=str, default='')
    parser.add_argument("--ref", type=str, default='')
    parser.add_argument("--write_path", type=str, default='')
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--do_sample", type=str2bool, default=False, help="Whether to use sampling")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--half_precision", type=str2bool, default=True)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--top_k", type=int, default=1)

    parser.add_argument("--mt", type=str, default='')
    parser.add_argument("--lang", type=str, default='zhen')
    parser.add_argument("--lora_path", type=str, default='')
    # 解析参数
    args = parser.parse_args()
    
    run_base_vllm(
        args.model_path,
        args.seed,
        args.src,
        args.ref,
        args.write_path,
        args.batch_size,
        args.top_k,
        args.max_new_tokens,
        args.top_p,
        args.temperature,
        args.tensor_parallel_size,
        args.half_precision,
        args.lang,
        args.lora_path
    )
    
if __name__ == "__main__":
    main()


