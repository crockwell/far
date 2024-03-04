#!/bin/bash

# step 0: train backbone
# the backbone we load from has already performed this!
# "weights/mapfree/rot6d_trans.ckpt"

# step 1: FAR: Transformer T_t
python train.py config/regression/mapfree/rot6d_trans_with_loftr.yaml \
    config/mapfree.yaml \
    --gpus 4 \
    --not_strict \
    --resume weights/mapfree/rot6d_trans.ckpt \
    --experiment far_transformer \
    --use_vanilla_transformer --d_model 256 

# step 2: FAR: One Round
python train.py config/regression/mapfree/rot6d_trans_with_loftr.yaml \
    config/mapfree.yaml \
    --gpus 4 \
    --not_strict \
    --resume weights/far_transformer/version_0/checkpoint/last.ckpt \
    --use_loftr_preds \
    --experiment far_loftr_one_round \
    --use_vanilla_transformer --d_model 256 

# step 3: FAR: Full
python train.py config/regression/mapfree/rot6d_trans_with_loftr.yaml \
    config/mapfree.yaml \
    --gpus 4 \
    --use_prior \
    --resume weights/far_loftr_one_round/version_0/checkpoint/last.ckpt \
    --use_loftr_preds \
    --experiment far_loftr \
    --use_vanilla_transformer --d_model 256 

# validate T
python submission.py \
    config/regression/mapfree/rot6d_trans_with_loftr.yaml \
    --checkpoint weights/far_loftr/version_0/checkpoints/last.ckpt \
    -o results/far_loftr_val \
    --use_loftr_preds --use_prior \
    --split val --use_vanilla_transformer --d_model 256 

python -m benchmark.mapfree \
    results/far_loftr_val/submission.zip \
    --split val
