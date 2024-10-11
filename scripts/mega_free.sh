#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_other.py \
    --root_dir ./data/free_dataset/grass --dataset_type colmap --dataset_name free_dataset --exp_name mega_size2 \
    --scene_name grass --downsample 0.5 \
    --num_epochs 20 --batch_size 4096 --lr 1e-2 --scale 16 --eval_lpips --gpu_id 0 \
    --moe_training --model_zoo_size 2 --model_type mega & \

CUDA_VISIBLE_DEVICES=1 python train_other.py \
    --root_dir ./data/free_dataset/hydrant --dataset_type colmap --dataset_name free_dataset --exp_name mega_size2 \
    --scene_name hydrant --downsample 0.5 \
    --num_epochs 20 --batch_size 4096 --lr 1e-2 --scale 64 --eval_lpips --gpu_id 1 \
    --moe_training --model_zoo_size 2 --model_type mega & \

CUDA_VISIBLE_DEVICES=2 python train_other.py \
    --root_dir ./data/free_dataset/lab --dataset_type colmap --dataset_name free_dataset --exp_name mega_size2 \
    --scene_name lab --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 16 --eval_lpips --gpu_id 2 \
    --moe_training --model_zoo_size 2 --model_type mega & \

CUDA_VISIBLE_DEVICES=3 python train_other.py \
    --root_dir ./data/free_dataset/pillar --dataset_type colmap --dataset_name free_dataset --exp_name mega_size2 \
    --scene_name pillar --downsample 0.5 \
    --num_epochs 20 --batch_size 4096 --lr 1e-2 --scale 64 --eval_lpips --gpu_id 3 \
    --moe_training --model_zoo_size 2 --model_type mega & \

CUDA_VISIBLE_DEVICES=4 python train_other.py \
    --root_dir ./data/free_dataset/road --dataset_type colmap --dataset_name free_dataset --exp_name mega_size2 \
    --scene_name road --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 16 --eval_lpips --gpu_id 4 \
    --moe_training --model_zoo_size 2 --model_type mega & \

CUDA_VISIBLE_DEVICES=5 python train_other.py \
    --root_dir ./data/free_dataset/sky --dataset_type colmap --dataset_name free_dataset --exp_name mega_size2 \
    --scene_name sky --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 16 --eval_lpips --gpu_id 5 \
    --moe_training --model_zoo_size 2 --model_type mega & \

CUDA_VISIBLE_DEVICES=6 python train_other.py \
    --root_dir ./data/free_dataset/stair --dataset_type colmap --dataset_name free_dataset --exp_name mega_size2 \
    --scene_name stair --downsample 0.5 \
    --num_epochs 20 --batch_size 4096 --lr 1e-2 --scale 64 --eval_lpips --gpu_id 6 \
    --moe_training --model_zoo_size 2 --model_type mega