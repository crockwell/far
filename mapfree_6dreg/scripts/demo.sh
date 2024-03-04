#!/bin/bash

python demo.py \
    config/regression/mapfree/rot6d_trans_with_loftr.yaml \
    --checkpoint pretrained_models/far_loftr.ckpt \
    --img_path0 demo/s00476_00000.jpg --img_path1 demo/s00476_00540.jpg \
    --k0 591.6007 591.6007 270.8742 351.5818 \
    --k1 587.2150 587.2150 269.4281 351.9291 

: '
-------------DEMO EXAMPLE------
val set idx 765, s00476/seq0/frame_00000.jpg s00476/seq1/frame_00540.jpg, 
Rotation mag 126, loftr err 145, reg err 37.3, FAR err 5.6, 
Translation mag 4.0, loftr err 4.3, reg err 1.6, FAR err 0.4

# results may vary from expected prediction by small margin (typically <0.001 per element)
expected prediction:
[-0.6417,  0.3945, -0.6577,  1.9560 ],
[-0.2080,  0.7359,  0.6443, -2.1584 ],
[ 0.7383,  0.5502, -0.3902,  3.1569 ]

gt:
[-0.5894,  0.3849, -0.7102,  2.0080],
[-0.2871,  0.7219,  0.6296, -1.8861],
[ 0.7551,  0.5750, -0.3150,  2.9398]
'
