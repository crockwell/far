#!/bin/bash

data_cfg_path="configs/data/mp3d.py"
main_cfg_path="configs/loftr/indoor/loftr_ds_dense.py"

# pretrained model
ckpt_path="pretrained_models/far_8pt.ckpt"
echo "running demo!"

python demo.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --img_path0 demo/mp3d_Vt2qJdWjCF2_0_12_2.png --img_path1 demo/mp3d_Vt2qJdWjCF2_0_12_32.png \
    --fx 517.97 --fy 517.97 --cx 320 --cy 240 --h 480 --w 640

: '
expected prediction:
[ 0.0367 -0.1899  0.9811 -0.4666]
[ 0.1938  0.9645  0.1795 -0.1628]
[-0.9803  0.1836  0.0722  0.8693]

gt:
[ 0.0322, -0.1907,  0.9811, -2.2427],
[ 0.1907,  0.9648,  0.1813, -0.7875],
[-0.9811,  0.1813,  0.0675,  4.1945]
'