import torch
import torch.nn.functional as F

class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x) 