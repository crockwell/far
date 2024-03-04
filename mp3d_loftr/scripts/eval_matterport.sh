#!/bin/bash

data_cfg_path="configs/data/mp3d.py"
main_cfg_path="configs/loftr/indoor/loftr_ds_dense.py"

# pretrained model
ckpt_path="pretrained_models/far_8pt.ckpt"

profiler_name="inference"
n_nodes=1  # mannually keep this the same with --nodes
n_gpus_per_node=10
torch_num_workers=0
batch_size=1  # per gpu

eval_split="test"
echo "evaluating on ${eval_split} split!!!"

python -u ./test.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} \
    --profiler_name=${profiler_name} \
    --benchmark \
    --eval_split=${eval_split} \
    --regress_rt  \
    --use_pos_embedding \
    --regress_loftr_layers 1 \
    --use_simple_moe \
    --regress_use_num_corres \
    --use_2wt \
    --num_coarse_loftr_layers 3 \
    --scale_8pt \
    --solver prior_ransac \
    --fine_pred_steps 2 \
    --use_many_ransac_thr 
