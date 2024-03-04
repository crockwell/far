#!/bin/bash

# step 0: train backbone
# the backbone we load from has already performed this!
# "weights/mapfree/rot6d_trans.ckpt"

# step 1: FAR: Transformer T_t
# note: this step is same as step 1 from train_mapfree_loftr
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
    --use_superglue_preds \
    --experiment far_superglue_one_round \
    --use_vanilla_transformer --d_model 256

# step 3: FAR: Full
python train.py config/regression/mapfree/rot6d_trans_with_loftr.yaml \
    config/mapfree.yaml \
    --gpus 4 \
    --use_prior \
    --resume weights/far_superglue_one_round/version_0/checkpoints/last.ckpt \
    --use_superglue_preds \
    --experiment far_superglue \
    --use_vanilla_transformer --d_model 256 

# validate
python submission.py \
    config/regression/mapfree/rot6d_trans_with_loftr.yaml \
    --checkpoint weights/far_superglue/version_0/checkpoints/last.ckpt \
    -o results/far_superglue \
    --use_superglue_preds --use_prior \
    --split val --use_vanilla_transformer --d_model 256 

python -m benchmark.mapfree \
    results/far_superglue/submission.zip \
    --split val
