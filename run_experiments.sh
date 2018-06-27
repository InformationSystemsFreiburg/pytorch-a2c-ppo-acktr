 python main.py \
   --env-name "ng_Worker" \
   --algo ppo \
   --use-gae \
   --lr 2.5e-4 \
   --clip-param 0.1 \
   --value-loss-coef 1 \
   --num-processes 8 \
   --num-steps 128 \
   --num-mini-batch 4 \
   --vis-interval 1 \
   --log-interval 1 \
   --ppo-epoch 10