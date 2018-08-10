#!/bin/bash

python ray_main.py \
   --env-name "ng_Worker" \
   --algo ppo \
   --use-gae \
   --lr 2.5e-4 \
   --clip-param 0.1 \
   --value-loss-coef 1 \
   --num-frames 1456000 \
   --num-processes 1 \
   --num-steps 14 \
   --num-mini-batch 14 \
   --vis-interval 1 \
   --log-interval 10 \
   --ppo-epoch 10 \
   --disable-env-normalize-ob \
   --disable-env-normalize-rw \
   --enable-0action-boost \
   --save-model-postfix "c9" \
   --log-dir "/tmp/gym/c9"

echo 'HEUREKA, I am done!'


