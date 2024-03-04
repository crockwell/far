#!/bin/bash

data_cfg_path="configs/data/mp3d.py"
main_cfg_path="configs/loftr/indoor/loftr_ds_dense.py"

echo "step 0: training lightweight LoFTR, which fits in 1080 Ti when combined with full model!"

n_nodes=1
n_gpus_per_node=10
torch_num_workers=12
batch_size=1
pin_memory=false
exp_name="loftr_lightweight"
num_epochs=100
ckpt_every_n_epochs=1

python -u ./train.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=100 \
    --flush_logs_every_n_steps=100 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --max_epochs=${num_epochs} \
    --ckpt_every_n_epochs=${ckpt_every_n_epochs} \
    --use_old_slow_warmup \
    --num_coarse_loftr_layers 3

echo "step 1: training FAR: Transformer!!!"

n_nodes=1
n_gpus_per_node=10
torch_num_workers=12
batch_size=1
pin_memory=false
exp_name="far_transformer"
ckpt_path="logs/tb_logs/loftr_lightweight/version_0/checkpoints/last.ckpt"

python -u ./train.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=100 \
    --flush_logs_every_n_steps=100 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --max_epochs=100 \
    --rt_weight_rot=0.01 \
    --rt_weight_tr=0.01 \
    --fine_weight=1.0 \
    --coarse_weight=1.0 \
    --regress_rt \
    --regress_loftr_layers 1 \
    --use_pos_embedding \
    --use_old_slow_warmup \
    --strict_false \
    --use_l1_rt_loss \
    --num_coarse_loftr_layers 3


echo "step 2: training FAR (One Round)!!!"

n_nodes=1
n_gpus_per_node=9
torch_num_workers=12
batch_size=1
pin_memory=false
exp_name="far_one_round"
ckpt_path="logs/tb_logs/far_transformer/version_0/checkpoints/last.ckpt"

python -u ./train.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=100 \
    --flush_logs_every_n_steps=100 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --max_epochs=100 \
    --rt_weight_rot=0.01 \
    --rt_weight_tr=0.01 \
    --fine_weight=1.0 \
    --coarse_weight=1.0 \
    --regress_rt \
    --regress_loftr_layers 1 \
    --use_pos_embedding \
    --use_old_slow_warmup \
    --strict_false \
    --use_simple_moe \
    --use_l1_rt_loss \
    --regress_use_num_corres \
    --use_2wt \
    --num_coarse_loftr_layers 3 \
    --scale_8pt


echo "step 3: training FAR (full)!!!"

n_nodes=1
n_gpus_per_node=9
torch_num_workers=12
batch_size=1
pin_memory=false
exp_name="far"
ckpt_path="logs/tb_logs/far_one_round/version_0/checkpoints/last.ckpt"

python -u ./train.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=100 \
    --flush_logs_every_n_steps=100 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --max_epochs=50 \
    --rt_weight_rot=0.01 \
    --rt_weight_tr=0.01 \
    --fine_weight=1.0 \
    --coarse_weight=1.0 \
    --regress_rt \
    --regress_loftr_layers 1 \
    --use_pos_embedding \
    --use_old_slow_warmup \
    --strict_false \
    --use_simple_moe \
    --use_l1_rt_loss \
    --regress_use_num_corres \
    --use_2wt \
    --num_coarse_loftr_layers 3 \
    --scale_8pt \
    --solver prior_ransac \
    --fine_pred_steps 2 \
    --use_one_cycle_lr \
    --use_many_ransac_thr \
    --lr 1e-5
