import jieba
import random
import os
import argparse
import json
import re
from src.utils.common import *
import sacrebleu

# 配置参数
parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str, default='aren', help='Language pair: aren, zhen, ende')
parser.add_argument('--model_ver', type=str, default='llama3', help='Model version: llama2, llama3')
parser.add_argument('--gpu', type=str, default='0', help='CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

# 设置 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# 路径映射配置
CONFIG = {
    'aren': {
        'train_src': '/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.ar',
        'train_ref': '/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.en',
        'test_ref': '/public/home/xiangyuduan/lyt/basedata/aren/test_ref.en',
        'prompt_template': 'Translate Arabic to English:\nArabic: {src}\nEnglish: {tgt}\n\n',
        'query_template': 'Translate Arabic to English:\nArabic: {src}\nEnglish:'
    },
    'zhen': {
        'train_src': '/public/home/xiangyuduan/lyt/basedata/125/train.ch',
        'train_ref': '/public/home/xiangyuduan/lyt/basedata/125/train.en',
        'test_ref': '/public/home/xiangyuduan/lyt/basedata/zhen/test_ref.en',
        'prompt_template': 'Translate Chinese to English:\nChinese: {src}\nEnglish: {tgt}\n\n',
        'query_template': 'Translate Chinese to English:\nChinese: {src}\nEnglish:'
    },
    'ende': {
        'train_src': '/public/home/xiangyuduan/lyt/basedata/459/train.en',
        'train_ref': '/public/home/xiangyuduan/lyt/basedata/459/train.de',
        'test_ref': '/public/home/xiangyuduan/lyt/basedata/ende/test_ref.de',
        'prompt_template': 'Translate English to German:\nEnglish: {src}\nGerman: {tgt}\n\n',
        'query_template': 'Translate English to German:\nEnglish: {src}\nGerman:'
    }
}

# 模型路径
MODELS = {
    'llama2': '/public/home/xiangyuduan/models/hf/Llama-2-7b-hf',
    'llama3': '/public/home/xiangyuduan/models/hf/Llama-3.1-8B'
}

# 1. 加载基础数据
print(f"Loading data for {args.lang} ...")
cfg = CONFIG[args.lang]
train_src = readline(cfg['train_src'])
train_ref = readline(cfg['train_ref'])
test_ref = readline(cfg['test_ref'])

# 确定数据目录 (aren2, aren3, zhen2...)
data_dir_name = f"{args.lang}{'2' if args.model_ver == 'llama2' else '3'}"
base_dir = f"/public/home/xiangyuduan/lyt/bad_word/key_data/{data_dir_name}"

# 输入文件：必须是 update_trigger_word.py 生成的带目标端 trigger_word 的文件
input_file_name = f"testbase_{args.model_ver}_tgt.json"
input_path = os.path.join(base_dir, input_file_name)

if not os.path.exists(input_path):
    print(f"Error: Input file not found: {input_path}")
    # 尝试在本地寻找（如果是本地测试环境）
    local_path = os.path.join(os.getcwd(), 'key_data', data_dir_name, input_file_name)
    if os.path.exists(local_path):
        print(f"Found local file: {local_path}")
        input_path = local_path
        base_dir = os.path.dirname(local_path)
    else:
        print("Cannot find input file. Exiting.")
        exit(1)

data = jsonreadline(input_path)

# 易错词索引文件：tgt_key_with_sent.json (由 tgt_badword_index.py 生成)
cfc_path = os.path.join(base_dir, "tgt_key_with_sent.json")
if not os.path.exists(cfc_path):
    print(f"Error: CFC file not found: {cfc_path}")
    exit(1)

cfc_data_list = jsonreadline(cfc_path)
# 转为字典加速查找
cfc_map = {item['trigger_word']: item for item in cfc_data_list}

baseline_mt = [i.get('test', '') if i.get('test') else i.get('mt', '') for i in data]

# 2. 加载模型
model_path = MODELS[args.model_ver]
print(f"Loading model from {model_path} ...")
try:
    tokenizer, model = load_vLLM_model(model_path, seed=0, tensor_parallel_size=1, half_precision=True)
except Exception as e:
    print(f"Failed to load model: {e}")
    print("If you are on Windows, vLLM is not supported. Please run this script on Linux with vLLM installed.")
    exit(1)

# 辅助函数
def random_pro():
    i = random.randint(0, len(train_src)-1)
    return cfg['prompt_template'].format(src=train_src[i], tgt=train_ref[i])

def find_cfc_item(word):
    return cfc_map.get(word)

# 3. 推理主循环
test_srcx = []
test_nmtx = []
test_refx = []

# 输出路径
output_filename = f"{args.lang}_{args.model_ver}_tgt_infer.json"
savepath = os.path.join(base_dir, output_filename)
print(f"Results will be saved to {savepath}")

# 记录处理后的数据
processed_data = []

for i in range(len(data)):
    item = data[i]
    trigger_word = item.get('trigger_word')
    
    mt_result = ""
    
    if trigger_word:
        # print(f"Trigger: {trigger_word}")
        # 查找包含该目标端易错词的示例
        cfc_item = find_cfc_item(trigger_word)
        
        exmall = []
        if cfc_item and 'index_list' in cfc_item and len(cfc_item['index_list']) > 0:
            indices = cfc_item['index_list'][:10] # 最多取10个
            
            for idx in indices:
                try:
                    s = cfg['prompt_template'].format(src=train_src[idx], tgt=train_ref[idx])
                    if len(s) <= 1000: # 简单的长度过滤
                        exmall.append(s)
                except IndexError:
                    continue
                    
        # 如果没有检索到，或者过滤后为空，使用随机
        if not exmall:
            exmall = [random_pro()]
            
        # 构造最终 Prompt
        # 这里只取第一个示例作为 Prompt (1-shot)
        # 如果需要多示例，可以 join
        # 假设这里是 1-shot
        prompt_content = exmall[0] 
        
        full_prompt = prompt_content + cfg['query_template'].format(src=item['src'])
        
        output = generate_with_vLLM_model(model, [full_prompt], n=1, stop=['\n'], top_p=0.00001, top_k=1, max_tokens=150, temperature=0)
        mt_result = output[0].outputs[0].text.strip()
        
        item['prompt'] = [prompt_content] # 记录使用的 prompt
        
    else:
        # 不触发时，使用 baseline 结果
        mt_result = baseline_mt[i]
        
    item['new_mt'] = mt_result # 暂存新翻译
    
    # 收集用于评估的数据
    test_srcx.append(item['src'])
    test_nmtx.append(mt_result)
    test_refx.append(test_ref[i])
    
    processed_data.append(item)

del model
import torch
torch.cuda.empty_cache()

# 4. 评估
print("Calculating metrics...")
try:
    comet_scores, avg_comet = count_comet(test_srcx, test_refx, test_nmtx)
    print(f"COMET: {avg_comet}")
    
    # COMET-Kiwi (Optional)
    try:
        cometfree_scores, avg_cometfree = count_comet(test_srcx, test_refx, test_nmtx, model_path='/public/home/xiangyuduan/models/hf/wmt22-cometkiwi-da/checkpoints/model.ckpt')
        print(f"COMET-Kiwi: {avg_cometfree}")
    except Exception as e:
        print(f"Skipping COMET-Kiwi: {e}")
        cometfree_scores = [0.0] * len(test_srcx)

except Exception as e:
    print(f"Error calculating COMET: {e}")
    comet_scores = [0.0] * len(test_srcx)
    cometfree_scores = [0.0] * len(test_srcx)

bleu = sacrebleu.corpus_bleu(test_nmtx, [test_refx]).score
print(f"BLEU: {bleu}")

# 5. 回填分数并保存
print(f"Saving final results to {savepath} ...")
with open(savepath, 'w', encoding='utf-8') as f:
    for i, item in enumerate(processed_data):
        item['comet'] = float(comet_scores[i]) if isinstance(comet_scores, list) else 0.0
        item['comet_free'] = float(cometfree_scores[i]) if isinstance(cometfree_scores, list) else 0.0
        item['mt'] = test_nmtx[i] # 更新最终翻译
        
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

print("Done.")
