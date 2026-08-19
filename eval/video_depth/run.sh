#!/usr/bin/env bash

set -euo pipefail

: "${CONDA_PREFIX:?activate the Conda environment before running video-depth evaluation}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

workdir='.'
model_name='ours'
datasets=('sintel' 'bonn' 'kitti')

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}_${model_name}"
    echo "$output_dir"
    torchrun --nproc-per-node=4 eval/video_depth/launch.py \
        --weights output_camera_prior/checkpoint-last.pth \
        --output_dir "$output_dir" \
        --eval_dataset "$data"
    python eval/video_depth/eval_depth.py \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --align "scale"
done
