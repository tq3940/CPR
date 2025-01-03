#!/bin/sh -x

# LightGCN
python src/main.py \
    --model_name LightGCN \
    --emb_size 128 \
    --n_layers 3 \
    --lr 1e-3 \
    --l2 1e-6 \
    --dataset Grocery_and_Gourmet_Food \
    --path data \
    --metric NDCG,HR \
    --topk 1,2,3,5,10,20 \
    --main_metric NDCG@2 \
    --batch_size 2048 \
    --gpu 0 \
    --test_epoch 20 \
    --num_workers 14

python src/main.py \
    --model_name LightGCN \
    --emb_size 128 \
    --n_layers 3 \
    --lr 1e-3 \
    --l2 1e-6 \
    --dataset MIND_small \
    --path data \
    --metric NDCG,HR \
    --topk 1,2,3,5,10,20 \
    --main_metric NDCG@2 \
    --batch_size 2048 \
    --gpu 0 \
    --test_epoch 20 \
    --num_workers 14

python src/main.py \
    --model_name LightGCN \
    --emb_size 128 \
    --n_layers 3 \
    --lr 1e-3 \
    --l2 1e-6 \
    --dataset MovieLens_1M \
    --path data \
    --metric NDCG,HR \
    --topk 1,2,3,5,10,20 \
    --main_metric NDCG@2 \
    --batch_size 2048 \
    --gpu 0 \
    --test_epoch 20 \
    --num_workers 14

