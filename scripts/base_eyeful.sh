
CUDA_VISIBLE_DEVICES=1 python train.py \
    --root_dir ./data/eyeful_tower/apartment --dataset_type eyeful --dataset_name eyeful_tower --exp_name base \
    --scene_name apartment --downsample 1 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 1 & \

CUDA_VISIBLE_DEVICES=2 python train.py \
    --root_dir ./data/eyeful_tower/office_view2 --dataset_type eyeful --dataset_name eyeful_tower --exp_name base \
    --scene_name office_view2 --downsample 1 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 2 & \

CUDA_VISIBLE_DEVICES=3 python train.py \
    --root_dir ./data/eyeful_tower/office1b --dataset_type eyeful --dataset_name eyeful_tower --exp_name base \
    --scene_name office1b --downsample 1 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 3 & \

CUDA_VISIBLE_DEVICES=4 python train.py \
    --root_dir ./data/eyeful_tower/riverview --dataset_type eyeful --dataset_name eyeful_tower --exp_name base \
    --scene_name riverview --downsample 1 \
    --num_epochs 20 --batch_size 8192 --lr 1e-2 --scale 4 --eval_lpips --gpu_id 4