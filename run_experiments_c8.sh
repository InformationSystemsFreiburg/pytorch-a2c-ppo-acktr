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
       --num-frames 2912000 \
       --num-processes 10 \
       --num-steps 14 \
       --num-mini-batch 364 \
       --vis-interval 1 \
       --log-interval 10 \
       --ppo-epoch 10 \
       --disable-env-normalize-ob \
       --disable-env-normalize-rw \
       --enable-0action-boost \
       --save-model-postfix "c8_w$nworker" \
       --log-dir "/tmp/gym/c8_w$nworker" \
       --number-of-workers $nworker
done

echo 'HEUREKA, I am done!'


