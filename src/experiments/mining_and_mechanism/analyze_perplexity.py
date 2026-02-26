from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

# 1. 加载模型和分词器
model_name = "/public/home/xiangyuduan/bli/blidata/models/hf/Qwen2.5-3B"  # 也可以用更小的 xglm-564M
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()


def compute_ppl(sentences):
    ppls = []
    for sent in sentences:
        # 编码
        enc = tokenizer(sent, return_tensors="pt")
        input_ids = enc["input_ids"].cuda()
        # 计算loss
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss.item()
        ppl = np.exp(loss)
        ppls.append(ppl)
    return ppls

if __name__ == "__main__":
    sentences = [
        "今天天气很好，我们去公园玩吧。",
        "机器翻译是自然语言处理的重要任务。",
        '＊w涣⒒】翟骸缕褝遍拥樽|600号（电话：二三八三三三一一）',
        '报名表格现于九龙旺角联运街三十号旺角政府合署地下油尖旺民政事务处咨询服务中心及油麻地众坊街六十号梁显利油麻地社区中心索取，截止报名日期为十一月七日。',
        '工商及科技局现正联同其他有关部门，根据新近取得的资料仔细研究在香l儗嵭小赋缕褝遍拥樽|600号（电话：二三八三三三一一）'
    ]
    ppls = compute_ppl(sentences)
    for sent, ppl in zip(sentences, ppls):
        print(f"句子: {sent}\n困惑度: {ppl:.2f}\n")