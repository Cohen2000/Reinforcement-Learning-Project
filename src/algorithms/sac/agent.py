# src/algorithms/sac/agent.py

"""
Architecture and hyperparameters got some inspiration from:
- Yarats & Kostrikov (2020): https://github.com/denisyarats/pytorch_sac
- Shyam (2021): https://github.com/pranz24/pytorch-soft-actor-critic
- Dai (2021): https://github.com/TianhongDai/reinforcement-learning-algorithms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch.distributions import Normal

def init_layer_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

class ActorNet(nn.Module):
    def __init__(self, obs_size, act_size, hidden=[256, 256], act_low=None, act_high=None, eps=1e-6):
        super(ActorNet, self).__init__()
        self.eps = eps
        self.fc1 = nn.Linear(obs_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.mu_head = nn.Linear(hidden[1], act_size)
        self.log_std_head = nn.Linear(hidden[1], act_size)
        self.apply(init_layer_weights)
        if act_low is not None and act_high is not None:
            self.register_buffer("scale", torch.FloatTensor((act_high - act_low) / 2.0))
            self.register_buffer("bias", torch.FloatTensor((act_high + act_low) / 2.0))
        else:
            self.scale = 1.0
            self.bias = 0.0

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=10)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        raw_action = dist.rsample() 
        tanh_action = torch.tanh(raw_action)
        action = tanh_action * self.scale + self.bias
        log_prob = dist.log_prob(raw_action) - torch.log(self.scale * (1 - tanh_action.pow(2)) + self.eps)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mu, std

class CriticNet(nn.Module):
    def __init__(self, obs_size, act_size, hidden=[256, 256]):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(obs_size + act_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.out = nn.Linear(hidden[1], 1)
        self.apply(init_layer_weights)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(xu))
        x = F.relu(self.fc2(x))
        q_val = self.out(x)
        return q_val

class ReplayMem:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, transition):
        self.buffer.append(transition)
    
    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, obs_size, action_space, config):
        self.obs_size = obs_size
        self.act_size = 4  
        self.device = config.get("device", torch.device("cpu"))
        self.gamma = config.get("gamma", 0.95)
        self.tau = config.get("tau", 0.005)
        self.batch_size = config.get("batch_size", 128)
        self.auto_entropy = config.get("auto_entropy", True)
        self.target_entropy = -float(self.act_size)
        self.alpha = config.get("alpha", 0.2)
        self.actor_lr = config.get("actor_lr", 1e-4)
        self.critic_lr = config.get("critic_lr", 1e-3)
        self.alpha_lr = config.get("alpha_lr", 1e-4)
        act_low = action_space.low[:self.act_size]
        act_high = action_space.high[:self.act_size]
        self.actor = ActorNet(obs_size, self.act_size, act_low=act_low, act_high=act_high).to(self.device)
        self.critic1 = CriticNet(obs_size, self.act_size).to(self.device)
        self.critic2 = CriticNet(obs_size, self.act_size).to(self.device)
        self.critic1_target = CriticNet(obs_size, self.act_size).to(self.device)
        self.critic2_target = CriticNet(obs_size, self.act_size).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=self.critic_lr)
        if self.auto_entropy:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=self.alpha_lr)
        self.replay = ReplayMem(capacity=config.get("replay_capacity", 100000))
        
    
    def act(self, state, eval_mode=False):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval_mode:
            with torch.no_grad():
                action, _, _, _ = self.actor.sample(state_tensor)
            return action.cpu().numpy()[0]
        else:
            action, _, _, _ = self.actor.sample(state_tensor)
            return action.cpu().detach().numpy()[0]

    def store_transition(self, transition):
        self.replay.add(transition)

    def update(self):
        if len(self.replay) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay.sample_batch(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones.astype(np.float32)).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_act, next_logp, _, _ = self.actor.sample(next_states)
            target_q1 = self.critic1_target(next_states, next_act)
            target_q2 = self.critic2_target(next_states, next_act)
            target_q_min = torch.min(target_q1, target_q2) - self.alpha * next_logp
            target = rewards + (1 - dones) * self.gamma * target_q_min

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        loss_q1 = F.mse_loss(current_q1, target)
        loss_q2 = F.mse_loss(current_q2, target)
        
        self.critic1_opt.zero_grad()
        loss_q1.backward()
        self.critic1_opt.step()
        
        self.critic2_opt.zero_grad()
        loss_q2.backward()
        self.critic2_opt.step()
        
        new_actions, logp, _, _ = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        loss_pi = (self.alpha * logp - q_new).mean()

        self.actor_opt.zero_grad()
        loss_pi.backward()
        self.actor_opt.step()

        if self.auto_entropy:
            loss_alpha = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            loss_alpha.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().detach()
        else:
            loss_alpha = torch.tensor(0.).to(self.device)

        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {"q1_loss": loss_q1.item(), "q2_loss": loss_q2.item(), "pi_loss": loss_pi.item(), "alpha_loss": loss_alpha.item()}

    def save(self, path):
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic1_opt": self.critic1_opt.state_dict(),
            "critic2_opt": self.critic2_opt.state_dict(),
            "alpha": self.alpha,
            "auto_entropy": self.auto_entropy,
        }
        if self.auto_entropy:
            checkpoint["log_alpha"] = self.log_alpha
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt"])
        self.critic1_opt.load_state_dict(checkpoint["critic1_opt"])
        self.critic2_opt.load_state_dict(checkpoint["critic2_opt"])
        self.alpha = checkpoint["alpha"]
        if self.auto_entropy and "log_alpha" in checkpoint and checkpoint["log_alpha"] is not None:
            self.log_alpha = checkpoint["log_alpha"]

