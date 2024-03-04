#!/bin/bash
export INTERIORNET_STREETLEARN_PATH=/home/cnris/vl/ExtremeRotation_code

# TRAINED
# CKPT=output/interiornet_t/checkpoints/120000.pth

# PRETRAINED
CKPT=pretrained_models/interiornet_t.pth

EXPNAME=interiornet_t_with_loftr
nice -n 19 python test_streetlearn_interiornet.py --exp ${EXPNAME} --transformer_depth 6 \
        --fusion_transformer --ckpt $CKPT \
        --datapath=$INTERIORNET_STREETLEARN_PATH --dataset interiornet --streetlearn_interiornet_type T --losson6d \
        --use_normalized_6d --use_loftr_gating \
        --from_saved_preds loftr_preds/interiornet