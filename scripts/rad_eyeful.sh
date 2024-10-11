
CUDA_VISIBLE_DEVICES=4 python train_ml.py \
    --root_dir ./data/eyeful_tower/apartment --dataset_type eyeful --dataset_name eyeful_tower --exp_name ours_size2 \
    --scene_name apartment --downsample 1 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 4 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 1e-4 --cv_loss_w 1e-2 & \

CUDA_VISIBLE_DEVICES=5 python train_ml.py \
    --root_dir ./data/eyeful_tower/office_view2 --dataset_type eyeful --dataset_name eyeful_tower --exp_name ours_size2 \
    --scene_name office_view2 --downsample 1 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 5 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 1e-4 --cv_loss_w 1e-2 & \

CUDA_VISIBLE_DEVICES=6 python train_ml.py \
    --root_dir ./data/eyeful_tower/office1b --dataset_type eyeful --dataset_name eyeful_tower --exp_name ours_size2 \
    --scene_name office1b --downsample 1 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 6 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 1e-4 --cv_loss_w 1e-2 & \

CUDA_VISIBLE_DEVICES=7 python train_ml.py \
    --root_dir ./data/eyeful_tower/riverview --dataset_type eyeful --dataset_name eyeful_tower --exp_name ours_size2 \
    --scene_name riverview --downsample 1 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 7 \
    --moe_training --model_zoo_size 2 --gate_type ray --depth_mutual_loss_w 1e-4 --cv_loss_w 1e-2