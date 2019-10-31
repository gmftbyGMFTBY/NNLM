#!/bin/bash
# Author: GMFTBY
# Time: 2019.10.21

# ./run.sh [train/vocab] xiaohuangji 0

mode=$1
dataset=$2
CUDA=$3

if [ $mode = 'vocab' ]; then
    python data_loader.py \
        --file ./data/$dataset/corpus.txt \
        --vocabp ./data/vocab.pkl \
        --datap ./data/$dataset/data.pkl \
        --maxsize 50000
        
elif [ $mode = 'train' ]; then
    rm ./ckpt/$dataset/*
    echo "[!] clear the checkpoints and begin to train"
    CUDA_VISIBLE_DEVICES="$CUDA" python train.py \
        --dataset $dataset \
        --lr 1e-3 \
        --grad_clip 5 \
        --batch-size 128 \
        --patience 5 \
        --epoch 20
        
elif [ $mode = 'pretrained' ]; then
    rm ./ckpt/$dataset/*
    
    echo "[!] process the dataset $dataset"
    python data_loader.py \
        --file ./data/$dataset/corpus.txt \
        --vocabp none \
        --datap ./data/$dataset/data.pkl \
        --maxsize 50000
        
    echo "[!] clear the checkpoints and begin to load and train the model"
    CUDA_VISIBLE_DEVICES="$CUDA" python train.py \
        --dataset $dataset \
        --lr 1e-5 \
        --grad_clip 5 \
        --batch-size 128 \
        --patience 5 \
        --epoch 20 \
        --mode pretrained 

else
    echo "[!] $mode is invalid, train/test mode is valid"
fi