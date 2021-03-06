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
       --num-frames 3920000 \
       --num-processes 14 \
       --num-steps 14 \
       --num-mini-batch 14 \
       --vis-interval 100 \
       --log-interval 100 \
       --ppo-epoch 10 \
       --disable-env-normalize-ob \
       --disable-env-normalize-rw \
       --enable-0action-boost \
       --recurrent-policy \
       --save-interval 10000 \
       --number-of-workers $nworker \
       --save-model-postfix "c11_w$nworker" \
       --log-dir "/tmp/gym/c11"
done

echo 'HEUREKA, I am done!'

# c7

#       --env-name "ng_Worker" \
#       --algo ppo \
#       --use-gae \
#       --lr 2.5e-4 \
#       --clip-param 0.1 \
#       --value-loss-coef 1 \
#       --num-frames 2912000 \
#       --num-processes 8 \
#       --num-steps 728 \
#       --num-mini-batch 8 \
#       --vis-interval 10 \
#       --log-interval 10 \
#       --ppo-epoch 10 \
#       --save-model-postfix "c7_w$nworker" \
#       --log-dir "/tmp/gym/c7_w$nworker" \
#       --disable-env-normalize-ob \
#       --disable-env-normalize-rw \
#       --number-of-workers $nworker \
#       --recurrent-policy
