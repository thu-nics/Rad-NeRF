#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train_ml.py \
    --root_dir ./data/llff/fern --dataset_type colmap --dataset_name llff --exp_name rad_size2_6views \
    --scene_name fern  --downsample 0.25 \
    --num_epochs 10 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 1 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 --num_view 6 & \

CUDA_VISIBLE_DEVICES=4 python train_ml.py \
    --root_dir ./data/llff/flower --dataset_type colmap --dataset_name llff --exp_name rad_size2_6views \
    --scene_name flower  --downsample 0.25 \
    --num_epochs 10 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 4 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 --num_view 6 & \

CUDA_VISIBLE_DEVICES=5 python train_ml.py \
    --root_dir ./data/llff/fortress --dataset_type colmap --dataset_name llff --exp_name rad_size2_6views \
    --scene_name fortress  --downsample 0.25 \
    --num_epochs 10 --batch_size 8192 --lr 1e-2 --scale 16 --eval_lpips --gpu_id 5 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 --num_view 6 & \

CUDA_VISIBLE_DEVICES=6 python train_ml.py \
    --root_dir ./data/llff/horns --dataset_type colmap --dataset_name llff --exp_name rad_size2_6views \
    --scene_name horns  --downsample 0.25 \
    --num_epochs 10 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 6 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 --num_view 6

CUDA_VISIBLE_DEVICES=1 python train_ml.py \
    --root_dir ./data/llff/leaves --dataset_type colmap --dataset_name llff --exp_name rad_size2_6views \
    --scene_name leaves  --downsample 0.25 \
    --num_epochs 10 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 1 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 --num_view 6 & \

CUDA_VISIBLE_DEVICES=4 python train_ml.py \
    --root_dir ./data/llff/orchids --dataset_type colmap --dataset_name llff --exp_name rad_size2_6views \
    --scene_name orchids  --downsample 0.25 \
    --num_epochs 10 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 4 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 --num_view 6 & \

CUDA_VISIBLE_DEVICES=6 python train_ml.py \
    --root_dir ./data/llff/room --dataset_type colmap --dataset_name llff --exp_name rad_size2_6views \
    --scene_name room  --downsample 0.25 \
    --num_epochs 10 --batch_size 8192 --lr 1e-2 --scale 8 --eval_lpips --gpu_id 0 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 --num_view 6 & \

CUDA_VISIBLE_DEVICES=6 python train_ml.py \
    --root_dir ./data/llff/trex --dataset_type colmap --dataset_name llff --exp_name rad_size2_6views \
    --scene_name trex  --downsample 0.25 \
    --num_epochs 10 --batch_size 8192 --lr 1e-2 --scale 8 --eval_lpips --gpu_id 6 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 --num_view 6 




# CUDA_VISIBLE_DEVICES=1 python train_ml.py \
#     --root_dir ./data/llff/fern --dataset_type colmap --dataset_name llff --exp_name rad_size2_3views \
#     --scene_name fern  --downsample 0.25 \
#     --num_epochs 10 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 1 \
#     --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 --num_view 3 & \ 

# CUDA_VISIBLE_DEVICES=4 python train_ml.py \
#     --root_dir ./data/llff/flower --dataset_type colmap --dataset_name llff --exp_name rad_size2_3views \
#     --scene_name flower  --downsample 0.25 \
#     --num_epochs 10 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 4 \
#     --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 --num_view 3 & \

# CUDA_VISIBLE_DEVICES=5 python train_ml.py \
#     --root_dir ./data/llff/fortress --dataset_type colmap --dataset_name llff --exp_name rad_size2_3views \
#     --scene_name fortress  --downsample 0.25 \
#     --num_epochs 10 --batch_size 8192 --lr 1e-2 --scale 16 --eval_lpips --gpu_id 5 \
#     --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 --num_view 3 & \

# CUDA_VISIBLE_DEVICES=6 python train_ml.py \
#     --root_dir ./data/llff/horns --dataset_type colmap --dataset_name llff --exp_name rad_size2_3views \
#     --scene_name horns  --downsample 0.25 \
#     --num_epochs 10 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 6 \
#     --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 --num_view 3

# CUDA_VISIBLE_DEVICES=1 python train_ml.py \
#     --root_dir ./data/llff/leaves --dataset_type colmap --dataset_name llff --exp_name rad_size2_3views \
#     --scene_name leaves  --downsample 0.25 \
#     --num_epochs 10 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 1 \
#     --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 --num_view 3 & \

# CUDA_VISIBLE_DEVICES=4 python train_ml.py \
#     --root_dir ./data/llff/orchids --dataset_type colmap --dataset_name llff --exp_name rad_size2_3views \
#     --scene_name orchids  --downsample 0.25 \
#     --num_epochs 10 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 4 \
#     --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 --num_view 3 & \

# CUDA_VISIBLE_DEVICES=5 python train_ml.py \
#     --root_dir ./data/llff/room --dataset_type colmap --dataset_name llff --exp_name rad_size2_3views \
#     --scene_name room  --downsample 0.25 \
#     --num_epochs 10 --batch_size 8192 --lr 1e-2 --scale 8 --eval_lpips --gpu_id 0 \
#     --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 --num_view 3 & \

# CUDA_VISIBLE_DEVICES=6 python train_ml.py \
#     --root_dir ./data/llff/trex --dataset_type colmap --dataset_name llff --exp_name rad_size2_3views \
#     --scene_name trex  --downsample 0.25 \
#     --num_epochs 10 --batch_size 8192 --lr 1e-2 --scale 8 --eval_lpips --gpu_id 6 \
#     --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 --num_view 3 