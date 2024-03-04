#!/bin/bash
export INTERIORNET_STREETLEARN_PATH=/home/cnris/vl/ExtremeRotation_code

# step 1: train FAR: Transformer
EXPNAME=interiornet_t_transformer
nice -n 19 python train.py --name ${EXPNAME} --gpus=1 --batch=6 \
        --lr=5e-4 --fusion_transformer --transformer_depth 6 \
        --w_tr 10 --w_rot 10 --steps 121000 \
        --datapath=$INTERIORNET_STREETLEARN_PATH --dataset interiornet --streetlearn_interiornet_type T --losson6d --use_normalized_6d

# step 2: train FAR: One Round
EXPNAME=interiornet_t
CKPT=output/interiornet_t_transformer/checkpoints/120000.pth
nice -n 19 python train.py --name ${EXPNAME} --gpus=1 --batch=6 \
        --lr=5e-4 --fusion_transformer --transformer_depth 6 \
        --w_tr 10 --w_rot 10 --steps 121000 \
        --datapath=$INTERIORNET_STREETLEARN_PATH --dataset interiornet --streetlearn_interiornet_type T \
        --losson6d --use_normalized_6d --use_loftr_gating --ckpt $CKPT \
        --from_saved_preds loftr_preds/interiornet

# step 3: w/ prior
# we can't train with prior, since no translation information on interiornet / streetlearn!
# so we skip this step