# Todo's

* idea: try configs with smaller `num_steps` configuration. This increases updates which could help to overcome the problem that agents learn to maintain every single day.
* change environment reward function: promote 0 action by setting reward to 1 if 0 was chosen and all machines are still running afterwards.


# General information
some required constraints:

`num-mini-batch <= num-processes * num-steps`: otherwise sampling will raise an exception


## Useful commands

Run a visdom server to monitor results:
```
nohup python -m visdom.server 1>visdom_out.log 2>visdom_err.log &
```

then, run firefox or chrom(ium) and connect to `http://localhost:8097`

list running processes

```
ps -aux | grep <...>
```


use `kill <process id>`  to shutdown visdom server.

hardcore version if nothing helps

```
kill -9 <process id>
```


To run experiments use the `run_experiments_<...>.sh` scripts.

* `run_experiments.sh`
* `run_experiments_rnn.sh`
* `run_experiments_feudal.sh`

To run rollouts use the `run_enjoy_<...>.sh` scripts.

* `run_enjoy.sh`
* `run_enjoy_rnn.sh`
* `run_enjoy_lstm.sh`

calculations for `num_updates`
```
num_updates = num_frames // num_steps // num_processes
```

```
num_frames = num_updates * num_steps * num_processes
```

for some example calculations sie excle file: `num_updates_calculations.xlxs`

some important paths:

* `./results`: default storing loaction for `enjoy` runs.
* `./trained_models`: default storing location for resulting models, trained during experiment execution
* `/tmp/gym/..`: default storing location for monitoring files. used by visdom to visualize training progress

# Experiment protocol

## currently running
* run_experiments_c8.sh -> cdsvmlinux
* run_experiments_ray.sh with c9 -> cdsvmlinux
* run_experiments_rnn.sh with c7 -> cdsvmlinux

## 2018-07-05
currently running on `cdsvmlinux`
c9: is the same as c8 but a ray version. highly parallelized
NON RNN and  0action_boost enabled
```
chmod 755 run_experiments_ray.sh
nohup ./run_experiments_ray.sh 1>run_experiments_ray_out.log 2>run_experiments_ray_err.log &
```


## 2018-07-04

**Training**
currently running on `cdsvmlinux`
c8 is the same es c5 with more `num_processes` and `lower num_steps`
NON RNN and  0action_boost enabled
```
chmod 755 run_experiments_c8.sh
nohup ./run_experiments_c8.sh 1>run_experiments_c8_out.log 2>run_experiments_c8_err.log &
```


currently running on `cdsvmlinux`
c9: is the same as c8 but a ray version. highly parallelized
NON RNN and  0action_boost enabled
```
chmod 755 run_experiments_ray.sh
nohup ./run_experiments_ray.sh 1>run_experiments_ray_out.log 2>run_experiments_ray_err.log &
```

## 2018-07-02

**Enjoy/Rollout**

run enjoy script for c5 config (PPO - NON RNN)
```
chmod 755 run_enjoy.sh
nohup ./run_enjoy.sh 1>run_enjoy_out.log 2>run_enjoy_err.log &
```


**Training**

run c7 on ng_Worker for all possible nworker configs.
PPO - RNN
```
chmod 755 run_experiments_rnn.sh
nohup ./run_experiments_rnn.sh 1>run_experiments_rnn_out.log 2>run_experiments_rnn_err.log &
```

## 2018-06-29
run c5 on ng_Worker for all possible nworker configs.
PPO - Non RNN
```
chmod 755 run_experiments.sh
nohup ./run_experiments.sh 1>run_experiments_out.log 2>run_experiments_err.log &
```

## 2018-06-28
currently running on lms: c5
PPO on Worker env without gru element.

### Results

-----

# Training configurations

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
   --disable-env-normalize-ob \
   --disable-env-normalize-rw \
   --save-model-postfix "c5" \
   --log-dir "/tmp/gym/c5" \
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

### visualize c6
```
nohup python main_visualize.py \
  --algo ppo \
  --env-name "ng_Worker" \
  --num-frames 364 \
  --log-dir "/tmp/gym/c6" \
  1>c6_vis_out.log 2>c6_vis_err.log &
```

## c7
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
   --num-mini-batch 8 \
   --vis-interval 10 \
   --log-interval 10 \
   --ppo-epoch 10 \
   --disable-env-normalize-ob \
   --disable-env-normalize-rw \
   --recurrent-policy \
   --save-model-postfix "c7_w$nworker" \
   --log-dir "/tmp/gym/c7_w$nworker" \
   --number-of-workers $nworker \
   1>c7_out.log 2>c7_err.log &
```

### visualize c7
```
nohup python main_visualize.py \
  --algo ppo \
  --env-name "ng_Worker" \
  --num-frames 2912000 \
  --log-dir "/tmp/gym/c7" \
  1>c7_vis_out.log 2>c7_vis_err.log &
```

## c8
like **c5** with smaller num_steps and 0action_boost enabled.
```
nohup python main.py \
   --env-name "ng_Worker" \
   --algo ppo \
   --use-gae \
   --lr 2.5e-4 \
   --clip-param 0.1 \
   --value-loss-coef 1 \
   --num-frames 2912000 \
   --num-processes 10 \
   --num-steps 14 \
   --num-mini-batch 14 \
   --vis-interval 1 \
   --log-interval 10 \
   --ppo-epoch 10 \
   --disable-env-normalize-ob \
   --disable-env-normalize-rw \
   --enable-0action-boost \
   --save-model-postfix "c8" \
   --log-dir "/tmp/gym/c8" \
   1>c8_out.log 2>c8_err.log &
```

### visualize c8
```
nohup python main_visualize.py \
  --algo ppo \
  --env-name "ng_Worker" \
  --num-frames 2912000 \
  --log-dir "/tmp/gym/c8" \
  1>c8_vis_out.log 2>c8_vis_err.log &
```

## c9
used for ray parallelization.
like **c8(c5)** with smaller num_steps and 0action_boost enabled.
```
nohup python main.py \
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
   --log-dir "/tmp/gym/c9" \
   1>c9_out.log 2>c9_err.log &
```

### visualize c9
```
nohup python main_visualize.py \
  --algo ppo \
  --env-name "ng_Worker" \
  --num-frames 1456000 \
  --log-dir "/tmp/gym/c9" \
  1>c9_vis_out.log 2>c9_vis_err.log &
```

## c10
used for ray parallelization.
like **c7(c5)**
PPO-RNN
```
nohup python ray_main.py \
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
   --recurrent-policy \
   --save-model-postfix "c10" \
   --log-dir "/tmp/gym/c10" \
   1>c10_out.log 2>c10_err.log &
```

### visualize c10
```
nohup python main_visualize.py \
  --algo ppo \
  --env-name "ng_Worker" \
  --num-frames 1456000 \
  --log-dir "/tmp/gym/c10" \
  1>c10_vis_out.log 2>c10_vis_err.log &
```

## c11
like **c8** with RNN, 0action boost,
```
nohup python main.py \
   --env-name "ng_Worker" \
   --algo ppo \
   --use-gae \
   --lr 2.5e-4 \
   --clip-param 0.1 \
   --value-loss-coef 1 \
   --num-frames 2912000 \
   --num-processes 10 \
   --num-steps 14 \
   --num-mini-batch 14 \
   --vis-interval 1 \
   --log-interval 10 \
   --ppo-epoch 10 \
   --disable-env-normalize-ob \
   --disable-env-normalize-rw \
   --enable-0action-boost \
   --recurrent-policy \
   --save-interval 10000 \
   --save-model-postfix "c11_w$nworker" \
   --log-dir "/tmp/gym/c11" \
   1>c11_out.log 2>c11_err.log &
```

### visualize c11
```
nohup python main_visualize.py \
  --algo ppo \
  --env-name "ng_Worker" \
  --num-frames 2912000 \
  --log-dir "/tmp/gym/c11" \
  1>c11_vis_out.log 2>c11_vis_err.log &
```


## Enjoy configurations
config id's match the config id's we used for training.

### c5
currently, results are stored in `./results/` and named `action_sequence_PPO-NON-RNN_w<>.csv` and `statistics_PPO-NON-RNN_w<>.csv`
renaming to PPO-NON-RNN-c5 happend afterwards and is not done on the server yet.
```
nohup python enjoy.py \
   --env-name "ng_Worker" \
   --path-to-ac "./trained_models/ppo/ng_Worker-ppo-c5_w$nworker-400.pt" \
   --log-interval 10 \
   --disable-env-normalize-ob \
   --disable-env-normalize-rw \
   --number-of-workers $nworker \
   --path-to-results-dir "./results/" \
   --strategy-name "PPO-NON-RNN-c5" \
   --number-of-episodes 100 \
   1>c5_out.log 2>c5_err.log &
```


### c7
currently, results are stored in `./results/`
```
nohup python enjoy.py \
   --env-name "ng_Worker" \
   --path-to-ac "./trained_models/ppo/ng_Worker-ppo-c7_w$nworker-400.pt" \
   --log-interval 10 \
   --disable-env-normalize-ob \
   --disable-env-normalize-rw \
   --number-of-workers $nworker \
   --path-to-results-dir "./results/" \
   --strategy-name "PPO-NON-RNN-c7" \
   --number-of-episodes 100 \
   1>c7_out.log 2>c7_err.log &
```

### c8
currently, results are stored in `./results/` and named `action_sequence_PPO-NON-RNN_w<>.csv` and `statistics_PPO-NON-RNN_w<>.csv`
renaming to PPO-NON-RNN-c5 happend afterwards and is not done on the server yet.
```
nohup python enjoy.py \
   --env-name "ng_Worker" \
   --path-to-ac "./trained_models/ppo/ng_Worker-ppo-c8_w$nworker-20700.pt" \
   --log-interval 10 \
   --disable-env-normalize-ob \
   --disable-env-normalize-rw \
   --number-of-workers $nworker \
   --path-to-results-dir "./results/" \
   --strategy-name "PPO-NON-RNN-c8" \
   --number-of-episodes 100 \
   1>c8_out.log 2>c8_err.log &
```