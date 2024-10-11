

CUDA_VISIBLE_DEVICES=1 python test.py \
    --root_dir ./data/llff/room --dataset_type colmap --dataset_name llff --exp_name rad_size2 \
    --scene_name room --downsample 0.25 --ckpt_path /share/guolidong-nfs/proj-nerf/rad-nerf/ckpts/llff/room/rad_size2/epoch=19.ckpt \
    --num_epochs 10 --batch_size 8192 --lr 1e-2 --scale 8 --eval_lpips --gpu_id 0 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2

# CUDA_VISIBLE_DEVICES=6 python train_ml.py \
#     --root_dir ./data/llff/room --dataset_type colmap --dataset_name llff --exp_name rad_size2_6views \
#     --scene_name room  --downsample 0.25 \
#     --num_epochs 10 --batch_size 8192 --lr 1e-2 --scale 8 --eval_lpips --gpu_id 0 \
#     --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 --num_view 6 & \

# CUDA_VISIBLE_DEVICES=6 python train_ml.py \
#     --root_dir ./data/llff/room --dataset_type colmap --dataset_name llff --exp_name rad_size2_6views \
#     --scene_name room  --downsample 0.25 \
#     --num_epochs 10 --batch_size 8192 --lr 1e-2 --scale 8 --eval_lpips --gpu_id 0 \
#     --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 5e-3 --cv_loss_w 1e-2 --num_view 6 & \