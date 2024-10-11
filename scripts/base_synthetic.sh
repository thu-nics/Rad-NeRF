#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train.py \
    --root_dir ./data/Synthetic_NeRF/Chair --dataset_type nsvf --dataset_name Synthetic_NeRF --exp_name base \
    --scene_name Chair  --downsample 1 \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --scale 0.5 --eval_lpips --gpu_id 1 & \

CUDA_VISIBLE_DEVICES=1 python train.py \
    --root_dir ./data/Synthetic_NeRF/Drums --dataset_type nsvf --dataset_name Synthetic_NeRF --exp_name base \
    --scene_name Drums  --downsample 1 \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --scale 0.5 --eval_lpips --gpu_id 1 & \

CUDA_VISIBLE_DEVICES=1 python train.py \
    --root_dir ./data/Synthetic_NeRF/Ficus --dataset_type nsvf --dataset_name Synthetic_NeRF --exp_name base \
    --scene_name Ficus  --downsample 1 \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --scale 0.5 --eval_lpips --gpu_id 2 & \

CUDA_VISIBLE_DEVICES=1 python train.py \
    --root_dir ./data/Synthetic_NeRF/Hotdog --dataset_type nsvf --dataset_name Synthetic_NeRF --exp_name base \
    --scene_name Hotdog  --downsample 1 \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --scale 0.5 --eval_lpips --gpu_id 3 & \

CUDA_VISIBLE_DEVICES=1 python train.py \
    --root_dir ./data/Synthetic_NeRF/Lego --dataset_type nsvf --dataset_name Synthetic_NeRF --exp_name base \
    --scene_name Lego  --downsample 1 \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --scale 0.5 --eval_lpips --gpu_id 4 & \

CUDA_VISIBLE_DEVICES=1 python train.py \
    --root_dir ./data/Synthetic_NeRF/Materials --dataset_type nsvf --dataset_name Synthetic_NeRF --exp_name base \
    --scene_name Materials  --downsample 1 \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --scale 0.5 --eval_lpips --gpu_id 5 & \

CUDA_VISIBLE_DEVICES=1 python train.py \
    --root_dir ./data/Synthetic_NeRF/Mic --dataset_type nsvf --dataset_name Synthetic_NeRF --exp_name base \
    --scene_name Mic  --downsample 1 \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --scale 0.5 --eval_lpips --gpu_id 6 & \

CUDA_VISIBLE_DEVICES=1 python train.py \
    --root_dir ./data/Synthetic_NeRF/Ship --dataset_type nsvf --dataset_name Synthetic_NeRF --exp_name base \
    --scene_name Ship  --downsample 1 \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --scale 0.5 --eval_lpips --gpu_id 7 