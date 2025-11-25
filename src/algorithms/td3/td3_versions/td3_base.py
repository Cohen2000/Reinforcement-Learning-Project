# Implemented by Luca Reichmann
# Used https://github.com/sfujim/TD3, https://github.com/XinJingHao/TD3-Pytorch as a reference (incl. hyperparameter choices) for the TD3 algorithm

import torch
import torch.nn as nn
from neural.feedforward import Feedforward
import pink

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
import utils.noise
import copy

class TargetNetContainer:
    def _copy_nets(self, orig_net, target_net):
        target_net.load_state_dict(orig_net.state_dict())

    def _soft_copy_nets(self, net, target_net, amount):
        for parameters, target_parameters in zip(net.parameters(), target_net.parameters()):
            target_parameters.data.copy_(amount * parameters.data + (1 - amount) * target_parameters.data)
        
class Actor(Feedforward):
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes, noisy):
        super(Actor, self).__init__(state_dim, hidden_sizes, action_dim, noisy, F.relu, F.tanh)
        self.max_action = max_action
		
    def forward(self, state):
        return self.max_action * super().forward(state)


class Critic(Feedforward):
    def __init__(self, state_dim, action_dim, hidden_sizes, noisy):
        super(Critic, self).__init__(state_dim + action_dim, hidden_sizes, 1, noisy, F.relu)

    def forward(self, observations, actions):
        input_tensor = torch.cat([observations, actions], dim=-1)
        return super().forward(input_tensor)


class Critics(TargetNetContainer):
    def __init__(self, observation_dim, action_dim, hidden_sizes, noisy):
        self.critic_1 = Critic(observation_dim, action_dim, hidden_sizes, noisy)
        self.critic_2 = Critic(observation_dim, action_dim, hidden_sizes, noisy)

        self.critic_1_target = Critic(observation_dim, action_dim, hidden_sizes, noisy)
        self.critic_2_target = Critic(observation_dim, action_dim, hidden_sizes, noisy)

        self._copy_nets(self.critic_1, self.critic_1_target)
        self._copy_nets(self.critic_2, self.critic_2_target)

    def copy_nets(self):
        self._copy_nets(self.critic_1, self.critic_1_target)
        self._copy_nets(self.critic_2, self.critic_2_target)

    def soft_copy_nets(self, amount):
        self._soft_copy_nets(self.critic_1, self.critic_1_target, amount)
        self._soft_copy_nets(self.critic_2, self.critic_2_target, amount)


class Actors(TargetNetContainer):
    def __init__(self, observation_dim, action_dim, max_action, hidden_sizes, noisy):
        self.actor = Actor(observation_dim, action_dim, max_action, hidden_sizes, noisy)
        self.actor_target = Actor(observation_dim, action_dim, max_action, hidden_sizes, noisy)

        self._copy_nets(self.actor, self.actor_target)

    def copy_nets(self):
        self._copy_nets(self.actor, self.actor_target)

    def soft_copy_nets(self, amount):
        self._soft_copy_nets(self.actor, self.actor_target, amount)


class TD3(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """
    def __init__(self, observation_space, action_space, num_actions, seq_len, **userconfig):        
        self._observation_space = observation_space
        self._obs_dim=self._observation_space.shape[0]
        self._action_space = action_space
        self._action_n = num_actions
        self._config = {
            "eps": 0.08,         
            "discount": 0.99,
            "buffer_size": int(1e6),
            "batch_size": 512,
            "learning_rate_actor": 1e-4,
            "learning_rate_critic": 1e-4,
            "hidden_sizes_actor": [256,256],
            "hidden_sizes_critic": [256,256],
            "policy_update_frequency": 2
        }
        self._config.update(userconfig)

        self._eps = self._config['eps']
        
        self.buffer = mem.Memory(max_size=self._config["buffer_size"])
        self.max_action = float(action_space.high[0])
        self.tau=0.005

        self.train_iter = 0
        self.loss = nn.MSELoss()
        self.action_noise = None

        self.critics_container = None
        self.actor_container = None

        self.critic_optimizer_1=None
        self.critic_optimizer_2=None
        self.actor_optimizer=None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def act(self, observation, eps=None):
        with torch.no_grad():
            if eps is None:
                eps = self._eps
            observation = torch.FloatTensor(observation).to(self.device)
            noise = self.action_noise()
            action = self.actor_container.actor(observation).cpu() + eps*noise
            action = action.clip(-self.max_action, self.max_action)
            return action

    def state(self):
        return (self.actor_container.actor.state_dict(), 
                self.critics_container.critic_1.state_dict(), 
                self.critics_container.critic_2.state_dict(),
                self.actor_optimizer.state_dict(),
                self.critic_optimizer_1.state_dict(),
                self.critic_optimizer_2.state_dict())

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def set_inference_mode(self, mode=True):
        if mode:
            self._eps = 0
            self.actor_container.actor.set_noisy(False)
            self.critics_container.critic_1.set_noisy(False)
            self.critics_container.critic_2.set_noisy(False)
        else:
            self._eps = 0.08
            self.actor_container.actor.set_noisy(True)
            self.critics_container.critic_1.set_noisy(True)
            self.critics_container.critic_2.set_noisy(True)

    def restore_state(self, state):
        self.actor_container.actor.load_state_dict(state[0])
        self.critics_container.critic_1.load_state_dict(state[1])
        self.critics_container.critic_2.load_state_dict(state[2])
        
        self.actor_container.copy_nets()
        self.critics_container.copy_nets()

        try:
            self.actor_optimizer.load_state_dict(state[3])
            self.critic_optimizer_1.load_state_dict(state[4])
            self.critic_optimizer_2.load_state_dict(state[5])
        except:
            print("Couldn't load optimizer states")

    def reset(self):
        self.action_noise.reset()
        self.actor_container.actor.reset()
        self.critics_container.critic_1.reset()
        self.critics_container.critic_2.reset()

    def train(self):
        critic_losses = []
        actor_losses = []
        self.train_iter+=1

        for i in range(32):
            with torch.no_grad():
                data = self.buffer.sample(self._config['batch_size'])
                states = torch.FloatTensor(np.stack(data[:,0])).to(self.device)
                actions = torch.FloatTensor(np.stack(data[:,1])).to(self.device)
                rewards = torch.FloatTensor(np.stack(data[:,2])[:,None]).to(self.device)
                new_states = torch.FloatTensor(np.stack(data[:,3])).to(self.device)
                is_done = torch.FloatTensor(np.stack(data[:,4])[:,None]).to(self.device)

                noise = self.action_noise()
                self.actor_container.actor_target.reset()
                new_actions = (self.actor_container.actor_target(new_states).cpu() + self._eps*noise)
                new_actions = new_actions.clip(-self.max_action, self.max_action).to(self.device)

                self.critics_container.critic_1_target.reset()
                self.critics_container.critic_2_target.reset()
                critic_1_Q = self.critics_container.critic_1_target(new_states.float(), new_actions.float())
                critic_2_Q = self.critics_container.critic_2_target(new_states.float(), new_actions.float())

                y = rewards + (1-is_done) * self._config["discount"] * torch.min(critic_1_Q, critic_2_Q)

            self.critics_container.critic_1.reset()
            self.critics_container.critic_2.reset()
            q_value_1 = self.critics_container.critic_1(states, actions)
            q_value_2 = self.critics_container.critic_2(states, actions)
            total_critic_loss = self.loss(y, q_value_1) + self.loss(y, q_value_2)

            self.critic_optimizer_1.zero_grad()
            self.critic_optimizer_2.zero_grad()
            total_critic_loss.backward()
            self.critic_optimizer_1.step()
            self.critic_optimizer_2.step()

            if self.train_iter % self._config["policy_update_frequency"] == 0:
                chosen_action = self.actor_container.actor(states)
                actor_loss = -self.critics_container.critic_1(states, chosen_action).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                with torch.no_grad():
                    self.critics_container.soft_copy_nets(self.tau)
                    self.actor_container.soft_copy_nets(self.tau)

                actor_losses.append(actor_loss.detach().cpu().numpy())

            critic_losses.append(total_critic_loss.detach().cpu().numpy())

        return critic_losses, actor_losses
