#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python train_other.py \
    --root_dir ./data/tanks_and_temples/tat_intermediate_M60 --dataset_type nerfpp --dataset_name tanks_and_temples --exp_name block_size2 \
    --scene_name M60 --downsample 1 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 2 \
    --moe_training --model_zoo_size 2 --model_type block & \

CUDA_VISIBLE_DEVICES=3 python train_other.py \
    --root_dir ./data/tanks_and_temples/tat_intermediate_Playground --dataset_type nerfpp --dataset_name tanks_and_temples --exp_name block_size2 \
    --scene_name Playground --downsample 1 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 8 --eval_lpips --gpu_id 3 \
    --moe_training --model_zoo_size 2 --model_type block & \

CUDA_VISIBLE_DEVICES=4 python train_other.py \
    --root_dir ./data/tanks_and_temples/tat_intermediate_Train --dataset_type nerfpp --dataset_name tanks_and_temples --exp_name block_size2 \
    --scene_name Train --downsample 1 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 32 --eval_lpips --gpu_id 4 \
    --moe_training --model_zoo_size 2 --model_type block & \

CUDA_VISIBLE_DEVICES=5 python train_other.py \
    --root_dir ./data/tanks_and_temples/tat_training_Truck --dataset_type nerfpp --dataset_name tanks_and_temples --exp_name block_size2 \
    --scene_name Truck --downsample 1 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 16 --eval_lpips --gpu_id 5 \
    --moe_training --model_zoo_size 2 --model_type block