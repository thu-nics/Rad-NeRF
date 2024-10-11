# !/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_other.py \
    --root_dir ./data/360_v2/bicycle --dataset_type colmap --dataset_name 360_v2 --exp_name switch_size2 \
    --scene_name bicycle --downsample 0.25 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 32 --eval_lpips --gpu_id 0 \
    --moe_training --model_zoo_size 2 --gate_type point --cv_loss_w 1e-4 --model_type switch & \

CUDA_VISIBLE_DEVICES=1 python train_other.py \
    --root_dir ./data/360_v2/bonsai --dataset_type colmap --dataset_name 360_v2 --exp_name switch_size2 \
    --scene_name bonsai --downsample 0.25 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 8 --eval_lpips --gpu_id 1 \
    --moe_training --model_zoo_size 2 --gate_type point --cv_loss_w 1e-4 --model_type switch & \

CUDA_VISIBLE_DEVICES=2 python train_other.py \
    --root_dir ./data/360_v2/counter --dataset_type colmap --dataset_name 360_v2 --exp_name switch_size2 \
    --scene_name counter --downsample 0.25 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 2 \
    --moe_training --model_zoo_size 2 --gate_type point --cv_loss_w 1e-4 --model_type switch & \

CUDA_VISIBLE_DEVICES=3 python train_other.py \
    --root_dir ./data/360_v2/garden --dataset_type colmap --dataset_name 360_v2 --exp_name switch_size2 \
    --scene_name garden --downsample 0.25 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 8 --eval_lpips --gpu_id 3 \
    --moe_training --model_zoo_size 2 --gate_type point --cv_loss_w 1e-4 --model_type switch & \

CUDA_VISIBLE_DEVICES=4 python train_other.py \
    --root_dir ./data/360_v2/kitchen --dataset_type colmap --dataset_name 360_v2 --exp_name switch_size2 \
    --scene_name kitchen --downsample 0.25 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 8 --eval_lpips --gpu_id 4 \
    --moe_training --model_zoo_size 2 --gate_type point --cv_loss_w 1e-4 --model_type switch & \

CUDA_VISIBLE_DEVICES=6 python train_other.py \
    --root_dir ./data/360_v2/room --dataset_type colmap --dataset_name 360_v2 --exp_name switch_size2 \
    --scene_name room --downsample 0.25 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 16 --eval_lpips --gpu_id 6 \
    --moe_training --model_zoo_size 2 --gate_type point --cv_loss_w 1e-4 --model_type switch & \

CUDA_VISIBLE_DEVICES=7 python train_other.py \
    --root_dir ./data/360_v2/stump --dataset_type colmap --dataset_name 360_v2 --exp_name switch_size2 \
    --scene_name stump --downsample 0.25 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 32 --eval_lpips --gpu_id 7 \
    --moe_training --model_zoo_size 2 --gate_type point --cv_loss_w 1e-4 --model_type switch