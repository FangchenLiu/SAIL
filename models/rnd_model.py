from torch import nn
import torch
from torch.nn import init
import numpy as np

class Coskx(nn.Module):
    def __init__(self, k=50):
        super(Coskx, self).__init__()
        self.k = k
    def forward(self, input):
        return torch.cos(input * self.k)

class RND(nn.Module):
    def __init__(self, d, last_size=512, ptb=10):
        super(RND, self).__init__()
        self.target = nn.Sequential(
            nn.Linear(d, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            Coskx(100),
            nn.Linear(512, last_size)
        )
        self.predictor = nn.Sequential(
            nn.Linear(d, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, last_size)
        )
        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, states):
        target_feature = self.target(states).detach()
        predict_feature = self.predictor(states)
        return ((predict_feature - target_feature) ** 2).mean(-1)

    def get_q(self, states):
        err = self.forward(states)
        return torch.exp(-10*err).detach()
