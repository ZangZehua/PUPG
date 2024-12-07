"""
This file defines all the neural network architectures available to use.
"""
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import torchvision


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class NonLinearQNet(nn.Module):
    def __init__(self, in_features, hidden_features, action_shape, v_min, v_max, n_atoms, noisy_std, log_softmax=False):
        super().__init__()
        self.action_shape = action_shape
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.value = nn.Sequential(
            NoisyLinear(in_features, hidden_features, noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_features, hidden_features, noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_features, n_atoms, noisy_std),
        )
        self.advantage = nn.Sequential(
            NoisyLinear(in_features, hidden_features, noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_features, hidden_features, noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_features, action_shape * n_atoms, noisy_std),
        )
        self.softmax = nn.LogSoftmax(dim=-1) if log_softmax else nn.Softmax(dim=-1)

    def forward(self, x, action=None):
        v = self.value(x)
        a = self.advantage(x)
        v, a = v.reshape(-1, 1, self.n_atoms), a.reshape(-1, self.action_shape, self.n_atoms)
        logit = v + a - a.mean(dim=1, keepdim=True)
        logit = self.softmax(logit)
        q_values = (logit * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, logit[torch.arange(len(x)), action]

    def get_q_dist(self, x):
        v = self.value(x)
        a = self.advantage(x)
        v, a = v.reshape(-1, 1, self.n_atoms), a.reshape(-1, self.action_shape, self.n_atoms)
        logit = v + a - a.mean(dim=1, keepdim=True)
        logit = self.softmax(logit)
        q_values = (logit * self.atoms).sum(2)
        return q_values

    def get_q_dist_action(self, x):
        v = self.value(x)
        a = self.advantage(x)
        v, a = v.reshape(-1, 1, self.n_atoms), a.reshape(-1, self.action_shape, self.n_atoms)
        logit = v + a - a.mean(dim=1, keepdim=True)
        logit = self.softmax(logit)
        q_values = (logit * self.atoms).sum(2)
        action = torch.argmax(q_values, 1)
        return q_values, action

    def reset_noise(self):
        for name, module in self.named_children():
            if module is not self.softmax:
                for layer in module:
                    if type(layer) is NoisyLinear:
                        layer.reset_noise()


class NatureCNN(nn.Module):
    """
    Implementation of the dueling architecture introduced in Wang et al. (2015).
    This implementation only works with a frame resolution of 84x84.
    """

    def __init__(self, depth):
        super().__init__()
        self.main = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=depth, out_channels=32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.out_features = 3136

    def forward(self, x):
        f = self.main(x)
        return f
