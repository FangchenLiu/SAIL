import torch
import torch.nn as nn
import numpy as np
from utils.tools import swish
from torch.nn import functional as F
import tqdm
import math

def weights_init_(m):
    if isinstance(m, nn.Linear):
        stdv = 1. / math.sqrt(m.weight.size(1))
        torch.nn.init.normal_(m.weight, std=1.0 / 2.0*stdv)
        torch.nn.init.constant_(m.bias, 0)

class ProbModel(nn.Module):
    def __init__(self, in_features, out_features, device, hidden_dim=400, fix_sigma=True, use_diag=True):
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        #define layers of shared feature, mean and variance
        self.layers = 3
        self.mean_layers = 1
        self.var_layers = 1

        self.affine = nn.ModuleList()
        self.mean = nn.ModuleList()

        self.use_diag = use_diag
        if use_diag == True:
            self.scale = nn.Parameter(torch.Tensor((0.5,)), requires_grad=True)
        else:
            self.scale = nn.Linear(hidden_dim, out_features)

        for i in range(self.layers):
            if i == 0:
                self.affine.append(nn.Linear(in_features,hidden_dim))
            else:
                self.affine.append(nn.Linear(hidden_dim,hidden_dim))

        for i in range(self.mean_layers):
            if i == self.mean_layers - 1:
                self.mean.append(nn.Linear(hidden_dim, out_features))
            else:
                self.mean.append(nn.Linear(hidden_dim, hidden_dim))

        '''
        for i in range(self.var_layers):
            if i == self.var_layers - 1:
                self.logvar.append(nn.Linear(hidden_dim, out_features))
            else:
                self.logvar.append(nn.Linear(hidden_dim, hidden_dim))
        '''

        self.apply(weights_init_)
        self.fix_sigma = fix_sigma
        #self.logvar = torch.log(nn.Parameter(torch.ones(1, out_features, dtype=torch.float32) * 0.1, requires_grad=True))
        if self.fix_sigma is not True:
            self.max_logvar = nn.Parameter(torch.ones(1, out_features, dtype=torch.float32) / 2.0)
            self.min_logvar = nn.Parameter(-torch.ones(1, out_features, dtype=torch.float32) * 10.0)

    def forward(self, inputs, ret_logvar=False):
        for affine in self.affine:
            inputs = swish(affine(inputs))

        mean = inputs
        for i, mean_layer in enumerate(self.mean):
            if i == len(self.mean) - 1:
                mean = mean_layer(mean)
            else:
                mean = swish(mean_layer(mean))
        if self.use_diag == True:
            logvar = torch.log((self.scale.expand(inputs.shape[0], self.out_features))).to(self.device)
        else:
            logvar = self.scale(inputs)
            #logvar = logvar.expand(inputs.shape[0], self.out_features).to(self.device)

        '''
        logvar = inputs
        for i, var_layer in enumerate(self.logvar):
            if i == len(self.logvar) - 1:
                logvar = var_layer(logvar)
            else:
                logvar = swish(var_layer(logvar))
        '''

        if self.fix_sigma == False:
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar
        return mean, torch.exp(logvar)

    def set_sigma(self, sigma):
        self.fix_sigma = sigma

class ForwardModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.affine_layers = nn.ModuleList()
        #self.bn_layers = nn.ModuleList()
        self.layers = 6
        self.first_layer = nn.Linear(self.in_features, hidden_dim)
        for i in range(self.layers):
            self.affine_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.relu = nn.ReLU()

        self.fc = nn.Linear(hidden_dim, out_features)
        self.apply(weights_init_)


    def forward(self, state, action):
        inputs = torch.cat((state, action), -1)
        last_output = self.relu(self.first_layer(inputs))
        for i, affine in enumerate(self.affine_layers):
            res = self.relu(affine(last_output))
            output = self.relu(last_output+res)
            last_output = output
        delta = self.fc(last_output)
        return delta

    def get_next_states(self, state, action):
        delta = self.forward(state, action)
        return state + delta

    def train(self, inputs_state, inputs_action, targets, optimizer, epoch=30, batch_size=256):
        #print('training model')
        idxs = np.arange(inputs_state.shape[0])
        np.random.shuffle(idxs)
        from tqdm import trange
        #epoch_range = trange(epoch, unit="epoch(s)", desc="Network training")
        num_batch = int(np.ceil(idxs.shape[-1] / batch_size))

        for _ in range(epoch):
            idxs = np.arange(inputs_state.shape[0])
            np.random.shuffle(idxs)
            for batch_num in range(num_batch):
                batch_idxs = idxs[batch_num * batch_size : (batch_num + 1) * batch_size]
                train_in_states = inputs_state[batch_idxs].float()
                train_in_actions = inputs_action[batch_idxs].float()
                train_targ = targets[batch_idxs].float()

                mean = self.forward(train_in_states, train_in_actions)
                train_losses = ((mean - train_targ) ** 2).mean()
                optimizer.zero_grad()
                train_losses.backward()
                optimizer.step()

        mean = self.forward(inputs_state, inputs_action)
        mse_losses = ((mean - targets) ** 2).mean(-1).mean(-1)
        print('forward model mse loss', mse_losses.detach().cpu().numpy())
        return mse_losses.detach().cpu().numpy()

class InverseModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.affine_layers = nn.ModuleList()
        #self.bn_layers = nn.ModuleList()
        self.layers = 6
        self.first_layer = nn.Linear(self.in_features, hidden_dim)
        for i in range(self.layers):
            self.affine_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.relu = nn.ReLU()

        self.final = nn.Linear(hidden_dim, out_features)
        self.apply(weights_init_)


    def forward(self, state, next_state):
        inputs = torch.cat((state, next_state), -1)
        last_output = self.relu(self.first_layer(inputs))
        for i, affine in enumerate(self.affine_layers):
            res = self.relu(affine(last_output))
            output = self.relu(last_output+res)
            last_output = output
        action = self.final(last_output)
        return action

    def train(self, state, next_state, actions, optimizer, epoch=30, batch_size=256):
        idxs = np.arange(state.shape[0])
        np.random.shuffle(idxs)
        num_batch = int(np.ceil(idxs.shape[-1] / batch_size))

        for _ in range(epoch):
            idxs = np.arange(state.shape[0])
            np.random.shuffle(idxs)
            for batch_num in range(num_batch):
                batch_idxs = idxs[batch_num * batch_size : (batch_num + 1) * batch_size]
                states_train = state[batch_idxs].float()
                next_state_train = next_state[batch_idxs].float()
                actions_targ = actions[batch_idxs].float()

                res = self.forward(states_train, next_state_train)
                train_losses = ((res - actions_targ) ** 2).mean()
                optimizer.zero_grad()
                train_losses.backward()
                optimizer.step()

        actions_pred = self.forward(state, next_state)
        mse_losses = ((actions_pred - actions) ** 2).mean(-1).mean(-1)
        print('inverse model mse loss', mse_losses.detach().cpu().numpy())
        return mse_losses.detach().cpu().numpy()