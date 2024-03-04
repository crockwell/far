#!/bin/bash

python submission.py \
    config/regression/mapfree/rot6d_trans_with_loftr.yaml \
    --checkpoint pretrained_models/far_loftr.ckpt \
    -o results/far_loftr_test \
    --use_loftr_preds --use_prior \
    --split test --use_vanilla_transformer --d_model 256 
