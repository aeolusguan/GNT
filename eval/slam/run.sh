#!/bin/bash

set -e

workdir='.'
model_name='ours'
datasets=('tum')


for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/slam/${data}_${model_name}"
    echo "$output_dir"
    torchrun --nproc-per-node=4 eval/slam/launch.py \
        --weights output_camera_prior/checkpoint-last.pth \
        --output_dir "$output_dir" \
        --eval_dataset "$data"
done