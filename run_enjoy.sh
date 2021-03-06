#!/bin/bash

for nworker in 1 10 20 30 40 50 60 70 80 90 100
do
    echo $nworker
    python enjoy.py \
        --env-name "ng_Worker" \
       --path-to-ac "./trained_models/ppo/ng_Worker-ppo-c11_w$nworker-final.pt" \
       --log-interval 10 \
       --disable-env-normalize-ob \
       --disable-env-normalize-rw \
       --number-of-workers $nworker \
       --path-to-results-dir "./results/" \
       --strategy-name "PPO-RNN-c11" \
       --number-of-episodes 100

done

echo 'HEUREKA, I am done!'

# c7
#    python enjoy.py \
#        --env-name "ng_Worker" \
#       --path-to-ac "./trained_models/ppo/ng_Worker-ppo-c7_w$nworker-400.pt" \
#       --log-interval 10 \
#       --disable-env-normalize-ob \
#       --disable-env-normalize-rw \
#       --number-of-workers $nworker \
#       --path-to-results-dir "./results/" \
#       --strategy-name "PPO-RNN-c7" \
#       --number-of-episodes 100

# c8
#    python enjoy.py \
#       --env-name "ng_Worker" \
#       --path-to-ac "./trained_models/ppo/ng_Worker-ppo-c8_w$nworker-20700.pt" \
#       --log-interval 10 \
#       --disable-env-normalize-ob \
#       --disable-env-normalize-rw \
#       --number-of-workers $nworker \
#       --path-to-results-dir "./results/" \
#       --strategy-name "PPO-NON-RNN-c8" \
#       --number-of-episodes 100