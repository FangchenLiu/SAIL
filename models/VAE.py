import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from torch.distributions import Normal
from models.ppo_models import weights_init_

MAX_LOG_STD = 0.5
MIN_LOG_STD = -20

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_size=128):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mean = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=MIN_LOG_STD, max=MAX_LOG_STD)
        return mean, log_std


class Decoder(torch.nn.Module):
    def __init__(self, input_dim, out_dim, hidden_size=128):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, out_dim)
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class VAE(torch.nn.Module):
    def __init__(self, state_dim, hidden_size=128, latent_dim=128):
        super(VAE, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = Encoder(state_dim, latent_dim=latent_dim, hidden_size=self.hidden_size)
        self.decoder = Decoder(latent_dim, state_dim, hidden_size=self.hidden_size)

    def forward(self, state):
        mu, log_sigma = self.encoder(state)
        sigma = torch.exp(log_sigma)
        sample = mu + torch.randn_like(mu)*sigma
        self.z_mean = mu
        self.z_sigma = sigma

        return self.decoder(sample)

    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)

    def get_next_states(self, states):
        mu, log_sigma = self.encoder(states)
        return self.decoder(mu)

    def get_loss(self, state, next_state):
        next_pred = self.get_next_states(state)
        return ((next_state-next_pred)**2).mean()

    def train(self, input, target, epoch, optimizer, batch_size=128, beta=0.1):
        idxs = np.arange(input.shape[0])
        np.random.shuffle(idxs)
        num_batch = int(np.ceil(idxs.shape[-1] / batch_size))
        for epoch in range(epoch):
            idxs = np.arange(input.shape[0])
            np.random.shuffle(idxs)
            for batch_num in range(num_batch):
                batch_idxs = idxs[batch_num * batch_size : (batch_num + 1) * batch_size]
                train_in = input[batch_idxs].float()
                train_targ = target[batch_idxs].float()
                optimizer.zero_grad()
                dec = self.forward(train_in)
                reconstruct_loss = ((train_targ-dec)**2).mean()
                ll = latent_loss(self.z_mean, self.z_sigma)
                loss = reconstruct_loss + beta*ll
                loss.backward()
                optimizer.step()
        val_input = input[idxs]
        val_target = target[idxs]
        val_dec = self.get_next_states(val_input)
        loss = ((val_target-val_dec)**2).mean().item()
        #print('vae loss', loss)
        return loss