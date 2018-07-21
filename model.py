import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import init, init_normc_


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, recurrent_policy):
        super(Policy, self).__init__()
        if len(obs_shape) == 3:
            self.base = CNNBase(obs_shape[0], recurrent_policy)
        elif len(obs_shape) == 1:
            self.base = MLPBase(obs_shape[0], recurrent_policy)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.state_size = self.base.state_size

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks):
        value, _, _ = self.base(inputs, states, masks)
        return value

    def evaluate_actions(self, inputs, states, masks, action):
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, states


class FeudalPolicy(nn.Module):
    def __init__(self, obs_shape, action_space, recurrent_policy):
        super(FeudalPolicy, self).__init__()

        self.num_inputs = obs_shape[0]

        if len(obs_shape) == 1:
            # extract worker input size -> bad practice, change this for production! Hard code 4 because we know that
            #  the underlaying environment returns the product of 100 machines times 4 sensor values. Deviding by 4
            # gives us the number of machines times the stacking factor.
            self.manager = ManagerBase(obs_shape[0], recurrent_policy)
            self.worker = WorkerBase(obs_shape[0] / 4, recurrent_policy)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.worker.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.worker.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.state_size = self.worker.state_size

        # build manager value function
        # build manager network
        # build worker value function
        # build worker network

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        # calculate manger output
        # calculate worker output
        # return worker logits
        pass

    def get_value(self, inputs, states, masks, action):
        pass

    def evaluate_actions(self, inputs, states, masks, action):
        pass

    def _build_manager(self):
        # calculate manger internal state
        ## input is a m x d matrix received from the environment
        ## flatten input to 1 x md vector
        ## input.view(input.size(0), -1)
        # calculate manager output g
        ## dilated rnn
        self.m_rnn = nn.LSTM(self.num_inputs, 64)
        self.hidden_g_hat = nn.Linear(64, 10)

        ## hidden g_hat
        ## g_hat
        ## g -> serves as input for worker
        ## store states c and h for input and output
        pass

    def _build_worker(self):
        # calculate w -> this requires g of the manager
        # calculate U -> worker lstm
        # calculate policy and sample
        pass


class ManagerBase(nn.Module):
    def __init__(self, num_inputs, g_dim, use_gru):
        super(ManagerBase, self).__init__()

        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))

        self.g_dim = g_dim
        self.hidden_dim = g_dim

        self.m_rnn = nn.LSTM(g_dim, g_dim)
        self.m_rnn_hidden = self._init_hidden()
        self.g = None

        self.m_v = None

        self.train()

    def _init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    @property
    def state_size(self):
        return self.g_dim

    @property
    def output_size(self):
        return 64

    def forward(self, inputs, states, masks):
        g_hat, self.m_rnn_hidden = self.m_rnn(inputs, self.m_rnn_hidden)
        g_hat_data = g_hat.data
        self.g = g_hat_data / torch.norm(g_hat_data, p=1, dim=1, keepdim=True)

        return self.m_v, self.g, self.m_rnn_hidden


class WorkerBase(nn.Module):
    def __init__(self, num_inputs, g_dim, use_gru):
        super(WorkerBase, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        self.g_dim = g_dim

        self.m_rnn = nn.LSTM(g_dim, g_dim)
        self.m_rnn_hidden = self._init_hidden()
        self.g = None

        self.m_v = None

        self.train()

    def _init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    @property
    def state_size(self):
        return self.g_dim

    @property
    def output_size(self):
        return 64

    def forward(self, inputs, states, masks):
        g_hat, self.m_rnn_hidden = self.m_rnn(inputs, self.m_rnn_hidden)
        g_hat_data = g_hat.data
        self.g = g_hat_data / torch.norm(g_hat_data, p=1, dim=1, keep_dim=True)

        return self.m_v, self.g, self.m_rnn_hidden

# class Policy(nn.Module):
#     def __init__(self, obs_shape, action_space, recurrent_policy):
#         super(Policy, self).__init__()
#         if len(obs_shape) == 3:
#             self.base = CNNBase(obs_shape[0], recurrent_policy)
#         elif len(obs_shape) == 1:
#             assert not recurrent_policy, \
#                 "Recurrent policy is not implemented for the MLP controller"
#             self.base = MLPBase(obs_shape[0])
#         else:
#             raise NotImplementedError
#
#         if action_space.__class__.__name__ == "Discrete":
#             num_outputs = action_space.n
#             self.dist = Categorical(self.base.output_size, num_outputs)
#         elif action_space.__class__.__name__ == "Box":
#             num_outputs = action_space.shape[0]
#             self.dist = DiagGaussian(self.base.output_size, num_outputs)
#         else:
#             raise NotImplementedError
#
#         self.state_size = self.base.state_size
#
#     def forward(self, inputs, states, masks):
#         raise NotImplementedError
#
#     def act(self, inputs, states, masks, deterministic=False):
#         value, actor_features, states = self.base(inputs, states, masks)
#         dist = self.dist(actor_features)
#
#         if deterministic:
#             action = dist.mode()
#         else:
#             action = dist.sample()
#
#         action_log_probs = dist.log_probs(action)
#         dist_entropy = dist.entropy().mean()
#
#         return value, action, action_log_probs, states
#
#     def get_value(self, inputs, states, masks):
#         value, _, _ = self.base(inputs, states, masks)
#         return value
#
#     def evaluate_actions(self, inputs, states, masks, action):
#         value, actor_features, states = self.base(inputs, states, masks)
#         dist = self.dist(actor_features)
#
#         action_log_probs = dist.log_probs(action)
#         dist_entropy = dist.entropy().mean()
#
#         return value, action_log_probs, dist_entropy, states


class CNNBase(nn.Module):
    def __init__(self, num_inputs, use_gru):
        super(CNNBase, self).__init__()

        init_ = lambda m: init(m,
                      nn.init.orthogonal_,
                      lambda x: nn.init.constant_(x, 0),
                      nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, 512)),
            nn.ReLU()
        )

        if use_gru:
            self.gru = nn.GRUCell(512, 512)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        init_ = lambda m: init(m,
          nn.init.orthogonal_,
          lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(512, 1))

        self.train()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    @property
    def output_size(self):
        return 512

    def forward(self, inputs, states, masks):
        x = self.main(inputs / 255.0)

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)

        return self.critic_linear(x), x, states


class MLPBase(nn.Module):
    def __init__(self, num_inputs, use_gru):
        super(MLPBase, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 64)),
            nn.Tanh()
        )

        if use_gru:
            self.gru = nn.GRUCell(64, 64)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 64)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(64, 1))

        self.train()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 64
        else:
            return 1

    @property
    def output_size(self):
        return 64

    def forward(self, inputs, states, masks):
        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                hidden_actor = states = self.gru(hidden_actor, states * masks)
            else:
                hidden_actor = hidden_actor.view(-1, states.size(0), hidden_actor.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(hidden_actor.size(0)):
                    hx = states = self.gru(hidden_actor[i], states * masks[i])
                    outputs.append(hx)
                hidden_actor = torch.cat(outputs, 0)

        return self.critic_linear(hidden_critic), hidden_actor, states



# class MLPBase(nn.Module):
#     def __init__(self, num_inputs):
#         super(MLPBase, self).__init__()
#
#         init_ = lambda m: init(m,
#               init_normc_,
#               lambda x: nn.init.constant_(x, 0))
#
#         self.actor = nn.Sequential(
#             init_(nn.Linear(num_inputs, 64)),
#             nn.Tanh(),
#             init_(nn.Linear(64, 64)),
#             nn.Tanh()
#         )
#
#         self.critic = nn.Sequential(
#             init_(nn.Linear(num_inputs, 64)),
#             nn.Tanh(),
#             init_(nn.Linear(64, 64)),
#             nn.Tanh()
#         )
#
#         self.critic_linear = init_(nn.Linear(64, 1))
#
#         self.train()
#
#     @property
#     def state_size(self):
#         return 1
#
#     @property
#     def output_size(self):
#         return 64
#
#     def forward(self, inputs, states, masks):
#         hidden_critic = self.critic(inputs)
#         hidden_actor = self.actor(inputs)
#
#         return self.critic_linear(hidden_critic), hidden_actor, states
