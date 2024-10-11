#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train.py \
    --root_dir ./data/free_dataset/grass --dataset_type colmap --dataset_name free_dataset --exp_name base \
    --scene_name grass --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 16 --eval_lpips --gpu_id 1 & \

CUDA_VISIBLE_DEVICES=1 python train.py \
    --root_dir ./data/free_dataset/hydrant --dataset_type colmap --dataset_name free_dataset --exp_name base \
    --scene_name hydrant --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 64 --eval_lpips --gpu_id 1 & \

CUDA_VISIBLE_DEVICES=1 python train.py \
    --root_dir ./data/free_dataset/lab --dataset_type colmap --dataset_name free_dataset --exp_name base \
    --scene_name lab --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 16 --eval_lpips --gpu_id 3 & \

CUDA_VISIBLE_DEVICES=1 python train.py \
    --root_dir ./data/free_dataset/pillar --dataset_type colmap --dataset_name free_dataset --exp_name base \
    --scene_name pillar --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 64 --eval_lpips --gpu_id 4 & \

CUDA_VISIBLE_DEVICES=1 python train.py \
    --root_dir ./data/free_dataset/road --dataset_type colmap --dataset_name free_dataset --exp_name base \
    --scene_name road --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 16 --eval_lpips --gpu_id 5 & \

CUDA_VISIBLE_DEVICES=1 python train.py \
    --root_dir ./data/free_dataset/sky --dataset_type colmap --dataset_name free_dataset --exp_name base \
    --scene_name sky --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 16 --eval_lpips --gpu_id 6 & \

CUDA_VISIBLE_DEVICES=1 python train.py \
    --root_dir ./data/free_dataset/stair --dataset_type colmap --dataset_name free_dataset --exp_name base \
    --scene_name stair --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 64 --eval_lpips --gpu_id 7