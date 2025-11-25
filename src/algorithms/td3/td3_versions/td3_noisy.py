import torch
import torch.nn as nn
from neural.feedforward import Feedforward
import pink
import td3_versions.td3_base as td3_base

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import optparse
import pickle

import utils.memory as mem
import utils.noise as noise
import copy


class TD3(td3_base.TD3):
    """
    Agent implementing Q-learning with NN function approximation.
    """
    def __init__(self, observation_space, action_space, num_actions, seq_len, **userconfig):
        super(TD3, self).__init__(observation_space, action_space, num_actions, seq_len, **userconfig)    
        self.action_noise = noise.OUNoise((self._action_n))

        self.critics_container = td3_base.Critics(observation_dim=self._obs_dim,
                                        action_dim=self._action_n,
                                        hidden_sizes=self._config['hidden_sizes_critic'], noisy=True)
        self.actor_container = td3_base.Actors(observation_dim=self._obs_dim,
                                        action_dim=self._action_n,
                                        max_action=self.max_action,
                                        hidden_sizes=self._config['hidden_sizes_actor'], noisy=True)

        self.critic_optimizer_1=torch.optim.Adam(self.critics_container.critic_1.parameters(),
                                        lr=self._config['learning_rate_critic'])
        self.critic_optimizer_2=torch.optim.Adam(self.critics_container.critic_2.parameters(),
                                        lr=self._config['learning_rate_critic'])
        self.actor_optimizer=torch.optim.Adam(self.actor_container.actor.parameters(),
                                        lr=self._config['learning_rate_actor'])

        