#!/bin/sh -x

# CPRMF
python src/main.py \
    --model_name CPRMF \
    --emb_size 128 \
    --lr 1e-3 \
    --l2 1e-6 \
    --dataset Grocery_and_Gourmet_Food \
    --path data \
    --metric NDCG,HR \
    --topk 1,2,3,5,10,20 \
    --main_metric NDCG@2 \
    --batch_size 2048 \
    --dyn_sample_rate 2 \
    --choose_rate 4 \
    --k_samples 2 \
    --num_workers 14

python src/main.py \
    --model_name CPRMF \
    --emb_size 128 \
    --lr 1e-3 \
    --l2 1e-6 \
    --dataset MIND_small \
    --path data \
    --metric NDCG,HR \
    --topk 1,2,3,5,10,20 \
    --main_metric NDCG@2 \
    --batch_size 2048 \
    --dyn_sample_rate 2 \
    --choose_rate 4 \
    --k_samples 2 \
    --num_workers 14

python src/main.py \
    --model_name CPRMF \
    --emb_size 128 \
    --lr 1e-3 \
    --l2 1e-6 \
    --dataset MovieLens_1M \
    --path data \
    --metric NDCG,HR \
    --topk 1,2,3,5,10,20 \
    --main_metric NDCG@2 \
    --batch_size 2048 \
    --dyn_sample_rate 2 \
    --choose_rate 4 \
    --k_samples 2 \
    --num_workers 14