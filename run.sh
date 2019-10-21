#!/bin/bash
# Author: GMFTBY
# Time: 2019.10.21

# ./run.sh [train/test] xiaohuangji 0

mode=$1
dataset=$2
CUDA=$3

if [ $mode = 'train' ]; then
    CUDA_VISIBLE_DEVICES="$CUDA" python train.py \
        --dataset $dataset \
        --lr 1e-3 \
        --grad_clip 5 \
        --batch-size 128 \
        --patience 10 \
        --epoch 20
elif [ $mode = 'test' ]; then
    CUDA_VISIBLE_DEVICES="$CUDA" python wrapper.py
else
    echo "[!] $mode is invalid, train/test mode is valid"
fi