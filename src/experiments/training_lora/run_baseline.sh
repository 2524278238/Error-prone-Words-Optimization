export CUDA_VISIBLE_DEVICES="0"


python baseline.py \
    --model_path /public/home/xiangyuduan/bli/blidata/models/hf/Llama-2-7b-hf \
    --seed 42 \
    --src /public/home/xiangyuduan/lyt/basedata/125/train.ch \
    --ref /public/home/xiangyuduan/lyt/basedata/125/train.en \
    --write_path /public/home/xiangyuduan/lyt/bad_word/key_data/zhen2/119w.en \
    --batch_size 16 \
    --num_beams 1 \
    --max_new_tokens 150 \
    --do_sample False \
    --temperature 0.0 \
    --top_p 0.00001 \
    --top_k 1 \