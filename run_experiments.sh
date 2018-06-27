nohup python -m visdom.server 1>visdom_out.log &

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
  nohup python main.py \
   --env-name "ng_Worker" \
   --algo ppo \
   --use-gae \
   --lr 2.5e-4 \
   --clip-param 0.1 \
   --value-loss-coef 1 \
   --num-frames 1500000 \
   --num-processes 8 \
   --num-steps 365 \
   --num-mini-batch 14 \
   --vis-interval 1 \
   --log-interval 10 \
   --ppo-epoch 32 \
   --save-model-postfix "c2" \
   1>c2_out.log 2>c2_err.log &

    # c3 - single round for testing
 nohup python main.py \
   --env-name "ng_Worker" \
   --algo ppo \
   --use-gae \
   --lr 2.5e-4 \
   --clip-param 0.1 \
   --value-loss-coef 1 \
   --num-frames 365 \
   --num-processes 1 \
   --num-steps 365 \
   --num-mini-batch 14 \
   --vis-interval 1 \
   --log-interval 10 \
   --ppo-epoch 32 \
   --save-model-postfix "c3" \
   1>c3_out.log 2>c3_err.log &
