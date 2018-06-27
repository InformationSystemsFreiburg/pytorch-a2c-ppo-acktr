# c1
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
   --ppo-epoch 10 \
   --save-model-postfix "c1"

  # c2
 python main.py \
   --env-name "ng_Worker" \
   --algo ppo \
   --use-gae \
   --lr 2.5e-4 \
   --clip-param 0.1 \
   --value-loss-coef 1 \
   --num-frames 1.5e6
   --num-processes 8 \
   --num-steps 365 \
   --num-mini-batch 14 \
   --vis-interval 1 \
   --log-interval 1 \
   --ppo-epoch 32 \
   --save-model-postfix "c2"
