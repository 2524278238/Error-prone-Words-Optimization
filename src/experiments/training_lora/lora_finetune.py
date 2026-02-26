# 基于目标端易错词采样的训练代码
# 适配多语向和多模型版本

from src.utils.common import *
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from datasets import Dataset
from trl import SFTTrainer
import os
import argparse
import random

# 禁用 WandB
os.environ["WANDB_DISABLED"] = "true"

# 配置
MODEL_PATHS = {
    "llama2": "/public/home/xiangyuduan/models/hf/Llama-2-7b-hf",
    "llama3": "/public/home/xiangyuduan/models/hf/Llama-3.1-8B" 
}

DATA_CONFIG = {
    'aren': {
        'prompt': 'Translate Arabic to English:\nArabic: {src}\nEnglish:{tgt}\n',
        'test_src': '/public/home/xiangyuduan/lyt/basedata/aren/test_src.ar',
        'test_ref': '/public/home/xiangyuduan/lyt/basedata/aren/test_ref.en'
    },
    'zhen': {
        'prompt': 'Translate Chinese to English:\nChinese: {src}\nEnglish:{tgt}\n',
        'test_src': '/public/home/xiangyuduan/lyt/basedata/zhen/test_src.zh',
        'test_ref': '/public/home/xiangyuduan/lyt/basedata/zhen/test_ref.en'
    },
    'ende': {
        'prompt': 'Translate English to German:\nEnglish: {src}\nGerman:{tgt}\n',
        'test_src': '/public/home/xiangyuduan/lyt/basedata/ende/test_src.en',
        'test_ref': '/public/home/xiangyuduan/lyt/basedata/ende/test_ref.de'
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Target-side Trigger Word Training")
    parser.add_argument("--lang", type=str, required=True, choices=["aren", "zhen", "ende"], help="Language pair")
    parser.add_argument("--model_ver", type=str, required=True, choices=["llama2", "llama3"], help="Model version")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    parser.add_argument("--batch_size", type=int, default=12, help="Per device train batch size")
    parser.add_argument("--grad_acc", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Num train epochs")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    print(f"Starting training for {args.lang} with {args.model_ver}...")
    
    # 1. 路径准备
    model_path = MODEL_PATHS[args.model_ver]
    
    # 训练数据路径 (由 sample_tgt_train_data.py 生成)
    # 格式: train_src_tgt_trigger_{lang}_{model_ver}.src
    data_dir = "/public/home/xiangyuduan/lyt/bad_word/train/train_data_tgt"
    suffix = f"tgt_trigger_{args.lang}_{args.model_ver}"
    train_src_path = os.path.join(data_dir, f"train_src_{suffix}.src")
    train_tgt_path = os.path.join(data_dir, f"train_ref_{suffix}.tgt")
    
    if not os.path.exists(train_src_path) or not os.path.exists(train_tgt_path):
        print(f"Error: Training data not found at {train_src_path} or {train_tgt_path}")
        exit(1)
        
    print(f"Loading training data from {train_src_path}...")
    src_lines = readline(train_src_path)
    tgt_lines = readline(train_tgt_path)
    
    # 测试集路径
    cfg = DATA_CONFIG[args.lang]
    print(f"Loading test data from {cfg['test_src']}...")
    test_src_lines = readline(cfg['test_src'])
    test_ref_lines = readline(cfg['test_ref'])
    
    # 2. 数据格式化
    prompt_template = cfg['prompt']
    
    def format_data(s_list, t_list):
        return [{'text': prompt_template.format(src=s, tgt=t)} for s, t in zip(s_list, t_list)]
        
    train_dict_list = format_data(src_lines, tgt_lines)
    test_dict_list = format_data(test_src_lines, test_ref_lines)
    
    train_dataset = Dataset.from_dict({key: [dic[key] for dic in train_dict_list] for key in train_dict_list[0]})
        # 验证集 (从测试集采样 1/10)
    val_dataset = Dataset.from_dict({key: [dic[key] for dic in test_dict_list] for key in test_dict_list[0]})
    validation_size = max(1, len(val_dataset) // 10)
    validation_indices = random.sample(range(len(val_dataset)), validation_size)
    val_dataset = val_dataset.select(validation_indices)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # 3. 模型加载
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Tokenizer 配置
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.eos_token = "</s>"
    tokenizer.bos_token = "<s>"
    
    # Model config
    model.config.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.generation_config.pad_token_id = 0
    model.generation_config.bos_token_id = 1
    model.generation_config.eos_token_id = 2

    
    # 4. LoRA 配置
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["gate_proj", "down_proj", "up_proj","q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model.enable_input_require_grads()
    
    # 5. 训练参数
    output_dir = f'/public/home/xiangyuduan/lyt/bad_word/train/models_tgt/{args.lang}_{args.model_ver}'
    
    training_arguments = TrainingArguments(
        load_best_model_at_end=True,
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        optim="adamw_torch",
        learning_rate=args.lr,
        eval_steps=200,
        save_steps=200,
        logging_steps=1, # 与 train.py 保持一致
        evaluation_strategy="steps",
        group_by_length=False,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.grad_acc,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        bf16=True, # 使用 BF16
        lr_scheduler_type="cosine",
        warmup_steps=0,
        save_total_limit=4, # 与 train.py 保持一致
    )
    
    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        peft_config=peft_config,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
    )
    
    # 7. 开始训练
    print("Starting training...")
    trainer.train()
    
    # 8. 保存
    print(f"Saving model to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")

if __name__ == "__main__":
    main()
