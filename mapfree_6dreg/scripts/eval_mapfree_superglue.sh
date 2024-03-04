#!/bin/bash

python submission.py \
    config/regression/mapfree/rot6d_trans_with_loftr.yaml \
    --checkpoint pretrained_models/far_superglue.ckpt \
    -o results/far_superglue_test \
    --use_superglue_preds --use_prior \
    --split test --use_vanilla_transformer --d_model 256 
