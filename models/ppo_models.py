import torch.nn as nn
import torch
from utils.math import *
from torch.distributions import Normal
from models.sac_models import weights_init_
from torch.distributions.categorical import Categorical

class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(256, 256), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.apply(weights_init_)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(256, 256, 256), activation='tanh', log_std=0):
        super().__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, x):
        action_mean, _, action_std = self.forward(x)
        normal = Normal(action_mean, action_std)
        action = normal.sample()
        return action

    def get_kl(self, x):
        mean1, log_std1, std1 = self.forward(x)

        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def get_log_prob_states(self, x):
        action_mean, action_log_std, action_std = self.forward(x)
        normal = Normal(action_mean, action_std)
        action = normal.rsample()
        entropy = normal.entropy().sum(-1)
        return action, entropy

    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}

    def get_entropy(self, x):
        mean, action_log_std, action_std = self.forward(x)
        dist = Normal(mean, action_std)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return entropy


class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_num, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        self.is_disc_action = True
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_head = nn.Linear(last_dim, action_num)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_prob = torch.softmax(self.action_head(x), dim=1)
        action = action_prob.multinomial(1)
        return action

    def log_prob(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        action_prob = torch.softmax(self.action_head(x), dim=1)
        return action_prob

    def select_action(self, x):
        action = self.forward(x)
        return action

    def get_kl(self, x):
        action_prob1 = self.log_prob(x)
        action_prob0 = action_prob1.detach()
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_prob = self.log_prob(x)
        return torch.log(action_prob.gather(1, actions.long().unsqueeze(1)))

    def get_log_prob_states(self, x):
        action_prob = self.log_prob(x)
        action = action_prob.multinomial(1)
        return action, torch.log(action_prob.gather(1, action.long().unsqueeze(1)))

    def get_fim(self, x):
        action_prob = self.log_prob(x)
        M = action_prob.pow(-1).view(-1).detach()
        return M, action_prob, {}

    def get_entropy(self, x):
        action_prob = self.log_prob(x)
        dist = Categorical(action_prob)
        entropy = dist.entropy()
        return entropy