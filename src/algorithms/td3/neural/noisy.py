# Implemented by Luca Reichmann
# Used https://github.com/alirezakazemipour/Rainbow, https://github.com/Kaixhin/Rainbow/blob/master/model.py as references for the implementation

import torch
import numpy as np
import math

class Noisy(torch.nn.Module):
    def __init__(self, in_dim, out_dim, stdev=0.5):
        super(Noisy, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activated_noise = True

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stdev = stdev

        self.weight_mean = torch.nn.Parameter(torch.empty(out_dim, in_dim, device=device))
        self.weight_stdev = torch.nn.Parameter(torch.empty(out_dim, in_dim, device=device))
        self.bias_mean = torch.nn.Parameter(torch.empty(out_dim, device=device))
        self.bias_stdev = torch.nn.Parameter(torch.empty(out_dim, device=device))

        self.register_buffer("w_eps", torch.empty(out_dim, in_dim, device=device))
        self.register_buffer("b_eps", torch.empty(out_dim, device=device))

        torch.nn.init.uniform_(self.weight_mean, -1.0 / np.sqrt(self.in_dim), 1.0 / np.sqrt(self.in_dim))
        torch.nn.init.uniform_(self.bias_mean, -1.0 / np.sqrt(self.in_dim), 1.0 / np.sqrt(self.in_dim))
        torch.nn.init.constant_(self.weight_stdev, self.stdev / np.sqrt(self.in_dim))
        torch.nn.init.constant_(self.bias_stdev, self.stdev / np.sqrt(self.out_dim))

        self.reset()

    def reset(self):
        x = np.random.standard_normal(self.in_dim)
        eps_in = torch.tensor(np.sign(x) * np.sqrt(np.abs(x)),
                              dtype=torch.float32, device=self.w_eps.device)
        eps_out = torch.tensor(np.random.standard_normal(self.out_dim), dtype=torch.float32, device=self.b_eps.device)

        self.register_buffer('w_eps', torch.outer(eps_out, eps_in))
        self.register_buffer('b_eps', eps_out)

    def forward(self, x):
        if self.activated_noise:
            weight = self.weight_mean + self.weight_stdev * self.w_eps
            bias = self.bias_mean + self.bias_stdev * self.b_eps
        else:
            weight = self.weight_mean
            bias = self.bias_mean
        return torch.nn.functional.linear(x, weight, bias)
