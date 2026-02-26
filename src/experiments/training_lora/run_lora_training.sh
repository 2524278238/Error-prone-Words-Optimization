#!/bin/bash
export LD_LIBRARY_PATH=/public/home/xiangyuduan/anaconda3/envs/rstar/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="0"
# 确保在当前目录下执行
cd "$(dirname "$0")"
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 定义输出目录
MODEL_BASE_DIR="/public/home/xiangyuduan/lyt/bad_word/train/models_tgt"
mkdir -p $MODEL_BASE_DIR

# 任务配置
#LANGS=("aren" "zhen" "ende")
LANGS=("zhen" "ende")
MODELS=("llama2" "llama3")
GPU_ID=0

echo "Start target-side trigger training..."

for lang in "${LANGS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "=================================================="
        echo "Training: Lang=$lang, Model=$model"
        echo "=================================================="
        
        # 检查数据是否存在
        DATA_FILE="/public/home/xiangyuduan/lyt/bad_word/train/train_data_tgt/train_src_tgt_trigger_${lang}_${model}.src"
        if [ ! -f "$DATA_FILE" ]; then
            echo "Warning: Data file $DATA_FILE not found. Skipping..."
            continue
        fi
        
        # 运行训练
        python train_tgt.py \
            --lang $lang \
            --model_ver $model \
            --gpu $GPU_ID \
            --batch_size 12 \
            --grad_acc 4 \
            --epochs 1
            
        if [ $? -ne 0 ]; then
            echo "Error: Training failed for $lang $model"
            exit 1
        fi
        
        echo "Finished $lang $model"
    done
done

echo "All training tasks completed."
