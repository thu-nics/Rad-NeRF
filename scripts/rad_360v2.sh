# !/bin/bash

CUDA_VISIBLE_DEVICES=1 python train_ml.py \
    --root_dir ./data/360_v2/bicycle --dataset_type 360v2 --dataset_name 360_v2 --exp_name rad_size2 \
    --scene_name bicycle --downsample 0.25 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 16 --eval_lpips --gpu_id 1 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 1e-4 --cv_loss_w 1e-2

CUDA_VISIBLE_DEVICES=1 python train_ml.py \
    --root_dir ./data/360_v2/bonsai --dataset_type 360v2 --dataset_name 360_v2 --exp_name rad_size2 \
    --scene_name bonsai --downsample 0.25 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 16 --eval_lpips --gpu_id 1 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 1e-4 --cv_loss_w 1e-2

CUDA_VISIBLE_DEVICES=1 python train_ml.py \
    --root_dir ./data/360_v2/counter --dataset_type 360v2 --dataset_name 360_v2 --exp_name rad_size2 \
    --scene_name counter --downsample 0.25 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 16 --eval_lpips --gpu_id 1 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 1e-4 --cv_loss_w 1e-2

CUDA_VISIBLE_DEVICES=1 python train_ml.py \
    --root_dir ./data/360_v2/garden --dataset_type 360v2 --dataset_name 360_v2 --exp_name rad_size2 \
    --scene_name garden --downsample 0.25 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 16 --eval_lpips --gpu_id 1 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 1e-4 --cv_loss_w 1e-2

CUDA_VISIBLE_DEVICES=1 python train_ml.py \
    --root_dir ./data/360_v2/kitchen --dataset_type 360v2 --dataset_name 360_v2 --exp_name rad_size2 \
    --scene_name kitchen --downsample 0.25 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 1 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 1e-4 --cv_loss_w 1e-2

CUDA_VISIBLE_DEVICES=1 python train_ml.py \
    --root_dir ./data/360_v2/room --dataset_type 360v2 --dataset_name 360_v2 --exp_name rad_size2 \
    --scene_name room --downsample 0.25 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 1 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 1e-4 --cv_loss_w 1e-2

CUDA_VISIBLE_DEVICES=1 python train_ml.py \
    --root_dir ./data/360_v2/stump --dataset_type 360v2 --dataset_name 360_v2 --exp_name rad_size2 \
    --scene_name stump --downsample 0.25 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 64 --eval_lpips --gpu_id 1 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 1e-4 --cv_loss_w 1e-2