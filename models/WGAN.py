import torch.nn as nn
import torch

class Generator(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size=400):
        super(Generator, self).__init__()
        main = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )
        self.main = main

    def forward(self, noise):
        output = self.main(noise)
        return output

class Discriminator(nn.Module):
    def __init__(self, num_inputs, layers=3, hidden_size=400, activation='tanh'):
        super(Discriminator, self).__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        self.num_layers = layers
        self.affine_layers = nn.ModuleList()

        for i in range(self.num_layers):
            if i == 0:
                self.affine_layers.append(nn.Linear(num_inputs, hidden_size))
            else:
                self.affine_layers.append(nn.Linear(hidden_size, hidden_size))

        self.logic = nn.Linear(hidden_size, 1)
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        prob = torch.sigmoid(self.logic(x))
        return prob

#for wgan: remove sigmoid in the last layer of discriminator
class W_Discriminator(nn.Module):

    def __init__(self, input_dim, hidden_size=400, layers=3):
        super(W_Discriminator, self).__init__()
        self.activation = torch.relu
        self.num_layers = layers
        self.affine_layers = nn.ModuleList()

        for i in range(self.num_layers):
            if i == 0:
                self.affine_layers.append(nn.Linear(input_dim, hidden_size))
            else:
                self.affine_layers.append(nn.Linear(hidden_size, hidden_size))

        self.final = nn.Linear(hidden_size, 1)
        self.final.weight.data.mul_(0.1)
        self.final.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        res = self.final(x)
        return res.view(-1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)