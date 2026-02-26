#!/bin/bash
# 顺序运行 6 个任务
# 使用: bash run_all_tgt.sh

# 默认使用 GPU 0，如果需要更改，请修改此处或直接运行 python 命令
export CUDA_VISIBLE_DEVICES=0

# AR-EN
echo "----------------------------------------"
echo "Running AR-EN Llama2..."
python run_tgt_bge.py --lang aren --model_ver llama2 --gpu 0

echo "----------------------------------------"
echo "Running AR-EN Llama3..."
python run_tgt_bge.py --lang aren --model_ver llama3 --gpu 0

# ZH-EN
echo "----------------------------------------"
echo "Running ZH-EN Llama2..."
python run_tgt_bge.py --lang zhen --model_ver llama2 --gpu 0

echo "----------------------------------------"
echo "Running ZH-EN Llama3..."
python run_tgt_bge.py --lang zhen --model_ver llama3 --gpu 0

# EN-DE
echo "----------------------------------------"
echo "Running EN-DE Llama2..."
python run_tgt_bge.py --lang ende --model_ver llama2 --gpu 0

echo "----------------------------------------"
echo "Running EN-DE Llama3..."
python run_tgt_bge.py --lang ende --model_ver llama3 --gpu 0

echo "----------------------------------------"
echo "All tasks completed."
