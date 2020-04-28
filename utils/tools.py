from os import path
import torch

def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../expert/assets'))

def swish(x):
    return x * torch.sigmoid(x)
