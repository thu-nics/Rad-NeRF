#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py \
    --root_dir ./data/scannet/scene0046_00 --dataset_type scannet --dataset_name scannet --exp_name base_2xmlp \
    --scene_name scene0046_00 --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 0 & \

CUDA_VISIBLE_DEVICES=1 python train.py \
    --root_dir ./data/scannet/scene0276_00 --dataset_type scannet --dataset_name scannet --exp_name base_2xmlp \
    --scene_name scene0276_00 --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 0 & \

CUDA_VISIBLE_DEVICES=2 python train.py \
    --root_dir ./data/scannet/scene0515_00 --dataset_type scannet --dataset_name scannet --exp_name base_2xmlp \
    --scene_name scene0515_00 --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 0 & \

CUDA_VISIBLE_DEVICES=3 python train.py \
    --root_dir ./data/scannet/scene0673_04 --dataset_type scannet --dataset_name scannet --exp_name base_2xmlp \
    --scene_name scene0673_04 --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 8 --eval_lpips --gpu_id 0