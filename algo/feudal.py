import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FEUDAL(object):
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None):
        pass

    def update(self, rollouts):
        pass