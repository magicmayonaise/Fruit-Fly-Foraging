
import torch
import torch.nn as nn

class OlfactoryCircuitNet(nn.Module):
    def __init__(self, connectome_weights, activation=nn.ReLU):
        super().__init__()
        self.W = nn.Parameter(connectome_weights, requires_grad=False)
        self.activation = activation()

    def forward(self, x):
        x = torch.matmul(x, self.W)
        x = self.activation(x)
        return x
