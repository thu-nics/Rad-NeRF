CUDA_VISIBLE_DEVICES=1 python train_ml.py \
    --root_dir ./data/Synthetic_NeRF/Chair --dataset_type nsvf --dataset_name Synthetic_NeRF --exp_name rad_size2 \
    --scene_name Chair  --downsample 1 \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --scale 0.5 --eval_lpips --gpu_id 1 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 0.005 --cv_loss_w 1e-2 & \

CUDA_VISIBLE_DEVICES=1 python train_ml.py \
    --root_dir ./data/Synthetic_NeRF/Drums --dataset_type nsvf --dataset_name Synthetic_NeRF --exp_name rad_size2 \
    --scene_name Drums  --downsample 1 \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --scale 0.5 --eval_lpips --gpu_id 1 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 0.005 --cv_loss_w 1e-2 & \

CUDA_VISIBLE_DEVICES=2 python train_ml.py \
    --root_dir ./data/Synthetic_NeRF/Ficus --dataset_type nsvf --dataset_name Synthetic_NeRF --exp_name rad_size2 \
    --scene_name Ficus  --downsample 1 \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --scale 0.5 --eval_lpips --gpu_id 2 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 0.005 --cv_loss_w 1e-2 & \

CUDA_VISIBLE_DEVICES=3 python train_ml.py \
    --root_dir ./data/Synthetic_NeRF/Hotdog --dataset_type nsvf --dataset_name Synthetic_NeRF --exp_name rad_size2 \
    --scene_name Hotdog  --downsample 1 \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --scale 0.5 --eval_lpips --gpu_id 3 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 0.005 --cv_loss_w 1e-2 & \

CUDA_VISIBLE_DEVICES=4 python train_ml.py \
    --root_dir ./data/Synthetic_NeRF/Lego --dataset_type nsvf --dataset_name Synthetic_NeRF --exp_name rad_size2 \
    --scene_name Lego  --downsample 1 \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --scale 0.5 --eval_lpips --gpu_id 4 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 0.005 --cv_loss_w 1e-2 & \

CUDA_VISIBLE_DEVICES=5 python train_ml.py \
    --root_dir ./data/Synthetic_NeRF/Materials --dataset_type nsvf --dataset_name Synthetic_NeRF --exp_name rad_size2 \
    --scene_name Materials  --downsample 1 \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --scale 0.5 --eval_lpips --gpu_id 5 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 0.005 --cv_loss_w 1e-2 & \

CUDA_VISIBLE_DEVICES=6 python train_ml.py \
    --root_dir ./data/Synthetic_NeRF/Mic --dataset_type nsvf --dataset_name Synthetic_NeRF --exp_name rad_size2 \
    --scene_name Mic  --downsample 1 \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --scale 0.5 --eval_lpips --gpu_id 6 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 0.005 --cv_loss_w 1e-2 & \

CUDA_VISIBLE_DEVICES=7 python train_ml.py \
    --root_dir ./data/Synthetic_NeRF/Ship --dataset_type nsvf --dataset_name Synthetic_NeRF --exp_name rad_size2 \
    --scene_name Ship  --downsample 1 \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --scale 0.5 --eval_lpips --gpu_id 7 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 0.005 --cv_loss_w 1e-2 