#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train_ml.py \
    --root_dir ./data/TanksAndTemple/Ignatius --dataset_type nsvf --dataset_name TanksAndTemple --exp_name rad_size5 \
    --scene_name Ignatius --downsample 1 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 0.5 --eval_lpips --gpu_id 1 \
    --moe_training --model_zoo_size 5 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 & \

CUDA_VISIBLE_DEVICES=2 python train_ml.py \
    --root_dir ./data/TanksAndTemple/Truck --dataset_type nsvf --dataset_name TanksAndTemple --exp_name rad_size5 \
    --scene_name Truck --downsample 1 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 0.5 --eval_lpips --gpu_id 2 \
    --moe_training --model_zoo_size 5 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 & \

CUDA_VISIBLE_DEVICES=3 python train_ml.py \
    --root_dir ./data/TanksAndTemple/Barn --dataset_type nsvf --dataset_name TanksAndTemple --exp_name rad_size5 \
    --scene_name Barn --downsample 1 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 0.5 --eval_lpips --gpu_id 3 \
    --moe_training --model_zoo_size 5 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 & \

CUDA_VISIBLE_DEVICES=4 python train_ml.py \
    --root_dir ./data/TanksAndTemple/Caterpillar --dataset_type nsvf --dataset_name TanksAndTemple --exp_name rad_size5 \
    --scene_name Caterpillar --downsample 1 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 0.5 --eval_lpips --gpu_id 4 \
    --moe_training --model_zoo_size 5 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 & \

CUDA_VISIBLE_DEVICES=5 python train_ml.py \
    --root_dir ./data/TanksAndTemple/Family --dataset_type nsvf --dataset_name TanksAndTemple --exp_name rad_size5 \
    --scene_name Family --downsample 1 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 0.5 --eval_lpips --gpu_id 5 \
    --moe_training --model_zoo_size 5 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2