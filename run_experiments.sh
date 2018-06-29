#!/bin/bash

for nworker in 1 10 20 30 40 50 60 70 80 90 100
do
    echo $nworker
    python main.py \
       --env-name "ng_Worker" \
       --algo ppo \
       --use-gae \
       --lr 2.5e-4 \
       --clip-param 0.1 \
       --value-loss-coef 1 \
       --num-frames 364 \
       --num-processes 1 \
       --num-steps 364 \
       --num-mini-batch 14 \
       --vis-interval 1 \
       --log-interval 1 \
       --ppo-epoch 32 \
       --save-model-postfix "c3_w$nworker" \
       --log-dir "/tmp/gym/c3_w$nworker" \
       --disable-env-normalize-ob \
       --disable-env-normalize-rw \
       --enable-debug-info-print \
       --number-of-workers $nworker
done

echo 'HEUREKA, I am done!'


