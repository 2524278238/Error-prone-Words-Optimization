#!/bin/bash
export LD_LIBRARY_PATH=/public/home/xiangyuduan/anaconda3/envs/rstar/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="0"
# 设置环境变量
export PYTHONPATH=/public/home/xiangyuduan/lyt/bad_word/train:$PYTHONPATH

# 定义共同参数
SRC_FILE="/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.ar"
TGT_FILE="/public/home/xiangyuduan/lyt/basedata/aren/WikiMatrix.ar-en.en"
TRIGGER_JSON="/public/home/xiangyuduan/lyt/bad_word/key_data/aren2/key_with_sent_comet.json"
OUTPUT_DIR="/public/home/xiangyuduan/lyt/bad_word/train/train_data/aren"
NUM_SAMPLES=16800

# #步骤1: 随机采样
# echo "第1步: 执行随机采样..."
# python sample_train_data.py \
#     --mode random \
#     --num $NUM_SAMPLES \
#     --src_file $SRC_FILE \
#     --tgt_file $TGT_FILE \
#     --output_dir $OUTPUT_DIR \
#     --seed 42

# # 步骤2: COMET高分采样
# echo "第2步: 执行COMET高分采样..."
# python sample_train_data.py \
#     --mode comet \
#     --num $NUM_SAMPLES \
#     --src_file $SRC_FILE \
#     --tgt_file $TGT_FILE \
#     --output_dir $OUTPUT_DIR

# 步骤3: 触发词采样（主试验）
echo "第3步: 执行触发词采样（主试验）..."
python sample_train_data.py \
    --mode trigger \
    --num $NUM_SAMPLES \
    --src_file $SRC_FILE \
    --tgt_file $TGT_FILE \
    --trigger_json $TRIGGER_JSON \
    --output_dir $OUTPUT_DIR

echo "所有采样任务完成！"
echo "输出文件位于: $OUTPUT_DIR"
echo "- 随机采样: train_src_random.ar, train_ref_random.en"
echo "- COMET高分采样: train_src_comet.ar, train_ref_comet.en"
echo "- 触发词采样（主试验）: train_src_trigger.ar, train_ref_trigger.en" 