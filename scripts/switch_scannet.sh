#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python train_other.py \
    --root_dir ./data/scannet/scene0046_00 --dataset_type scannet --dataset_name scannet --exp_name switch_size2 \
    --scene_name scene0046_00 --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 4 \
    --moe_training --model_zoo_size 2 --gate_type point --cv_loss_w 1e-4 --model_type switch & \

CUDA_VISIBLE_DEVICES=4 python train_other.py \
    --root_dir ./data/scannet/scene0276_00 --dataset_type scannet --dataset_name scannet --exp_name switch_size2 \
    --scene_name scene0276_00 --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 5 \
    --moe_training --model_zoo_size 2 --gate_type point --cv_loss_w 1e-4 --model_type switch & \

CUDA_VISIBLE_DEVICES=6 python train_other.py \
    --root_dir ./data/scannet/scene0515_00 --dataset_type scannet --dataset_name scannet --exp_name switch_size2 \
    --scene_name scene0515_00 --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 6 \
    --moe_training --model_zoo_size 2 --gate_type point --cv_loss_w 1e-4 --model_type switch & \

CUDA_VISIBLE_DEVICES=7 python train_other.py \
    --root_dir ./data/scannet/scene0673_04 --dataset_type scannet --dataset_name scannet --exp_name switch_size2 \
    --scene_name scene0673_04 --downsample 0.5 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 8 --eval_lpips --gpu_id 7 \
    --moe_training --model_zoo_size 2 --gate_type point --cv_loss_w 1e-4 --model_type switch