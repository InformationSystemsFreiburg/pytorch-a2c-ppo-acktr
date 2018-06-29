# General
Run a visdom server to monitor results:
```
nohup python -m visdom.server 1>visdom_out.log 2>visdom_err.log &
```

Process id of last run: 123579

use `killall <process id>`  to shutdown visdom server.

To run experiments use the `run_experiments.sh` script.

calculations for `num_updates`
```
num_updates = num_frames // num_steps // num_processes
```

```
num_frames = num_updates * num_steps * num_processes
```

for `num_updates = 500` ->

# Protocol

## 2018-06-29
run c5  on ng_Worker for all possible nworker configs.
PPO - Non RNN
```
chmod 755 run_experiments.sh
nohup run_experiments.sh 1>run_experiments_out.log 2>run_experiments_err.log &
```

## 2018-06-28
currently running on lms: c5
PPO on Worker env without gru element.

### Results

-----

# Configurations

`num_steps` 364 entspricht 1 Jahr. dann ist der letzte reward den wir erhalten der letzte tag des jahres.


## c1
```
nohup python main.py \
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
   --save-model-postfix "c1" \
   --log-dir "/tmp/gym/c1" \
   1>c1_out.log 2>c1_err.log &
 ```

### visualize c1
```
nohup python main_visualize.py \
  --algo ppo \
  --env-name "ng_Worker" \
  --log-dir "/tmp/gym/c1" \
  1>c1_vis_out.log 2>c1_vis_err.log &
```

## c2
```
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
   --log-dir "/tmp/gym/c2" \
   1>c2_out.log 2>c2_err.log &
 ```

### visualize c2
```
nohup python main_visualize.py \
  --algo ppo \
  --env-name "ng_Worker" \
  --num-frames 1500000 \
  --log-dir "/tmp/gym/c2" \
  1>c2_vis_out.log 2>c2_vis_err.log &
```

## c3 - single round for testing - No GRU
```
nohup python main.py \
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
   --save-model-postfix "c3" \
   --log-dir "/tmp/gym/c3" \
   --disable-env-normalize-ob \
   --disable-env-normalize-rw \
   --enable-debug-info-print \
   1>c3_out.log 2>c3_err.log &
 ```

### visualize c3
```
nohup python main_visualize.py \
  --algo ppo \
  --env-name "ng_Worker" \
  --num-frames 1500000 \
  --log-dir "/tmp/gym/c3" \
  1>c3_vis_out.log 2>c3_vis_err.log &
```

## c4
```
nohup python main.py \
   --env-name "ng_Worker" \
   --algo ppo \
   --use-gae \
   --lr 2.5e-4 \
   --clip-param 0.1 \
   --value-loss-coef 1 \
   --num-frames 1000000 \
   --num-processes 8 \
   --num-steps 364 \
   --num-mini-batch 14 \
   --vis-interval 10 \
   --log-interval 10 \
   --ppo-epoch 32 \
   --save-model-postfix "c4" \
   --log-dir "/tmp/gym/c4" \
   1>c4_out.log 2>c4_err.log &
 ```

### visualize c4
```
nohup python main_visualize.py \
  --algo ppo \
  --env-name "ng_Worker" \
  --num-frames 1000000 \
  --log-dir "/tmp/gym/c4" \
  1>c4_vis_out.log 2>c4_vis_err.log &
```

## c5
```
nohup python main.py \
   --env-name "ng_Worker" \
   --algo ppo \
   --use-gae \
   --lr 2.5e-4 \
   --clip-param 0.1 \
   --value-loss-coef 1 \
   --num-frames 2912000 \
   --num-processes 8 \
   --num-steps 728 \
   --num-mini-batch 364 \
   --vis-interval 1 \
   --log-interval 10 \
   --ppo-epoch 10 \
   --save-model-postfix "c5" \
   --log-dir "/tmp/gym/c5" \
   --disable-env-normalize-ob \
   --disable-env-normalize-rw \
   1>c5_out.log 2>c5_err.log &
```

### visualize c5
```
nohup python main_visualize.py \
  --algo ppo \
  --env-name "ng_Worker" \
  --num-frames 1000000 \
  --log-dir "/tmp/gym/c5" \
  1>c5_vis_out.log 2>c5_vis_err.log &
```

## c6 - single round for testing - GRU
```
nohup python main.py \
   --env-name "ng_Worker" \
   --algo ppo \
   --use-gae \
   --lr 2.5e-4 \
   --clip-param 0.1 \
   --value-loss-coef 1 \
   --num-frames 364 \
   --num-processes 1 \
   --num-steps 364 \
   --num-mini-batch 1 \
   --vis-interval 1 \
   --log-interval 1 \
   --ppo-epoch 32 \
   --save-model-postfix "c6" \
   --log-dir "/tmp/gym/c6" \
   --disable-env-normalize-ob \
   --disable-env-normalize-rw \
   --enable-debug-info-print \
   --recurrent-policy \
   1>c6_out.log 2>c6_err.log &
 ```

### visualize c3
```
nohup python main_visualize.py \
  --algo ppo \
  --env-name "ng_Worker" \
  --num-frames 364 \
  --log-dir "/tmp/gym/c6" \
  1>c6_vis_out.log 2>c6_vis_err.log &
```