OUTPUT_DIR=output
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29604 launch.py --datapath /mnt/DATA/mobilerobot/tguan/TartanAir \
    --pretrained="da3-base.pt" \
    --lr=0.00005 --min_lr=0.000001 --warmup_epochs=3 --epochs=40 --batch_size=1 --accum_iter=4 \
    --save_freq=1 --keep_freq=50 --amp=1 --n_frames 6 --edges 18 --accum_iter 2 \
    2>&1 | tee -a ${OUTPUT_DIR}/train.log