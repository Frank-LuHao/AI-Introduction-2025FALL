import torch.nn as nn
import torch.nn.functional as F
import torch as th

# QR_DQN 算法
class QR_DQN_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QR_DQN_net, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.seq(x)

# C51 算法
class C51_net(nn.Module):
    def __init__(self, input_dim, atom_num):
        super(C51_net, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, atom_num)
        )
    def forward(self, x):
        return F.softmax(self.seq(x), dim=-1)

# IQN 算法
class IQN_net(nn.Module):
    def __init__(self, input_dim):
        super(IQN_net, self).__init__()
        self.input_dim = input_dim
        self.phi = nn.Linear(self.input_dim, self.input_dim)
        self.seq = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, tau):
        quantile_embed = self.phi(th.cos(th.pi * th.arange(0, self.input_dim) * tau.unsqueeze(-1)))
        return self.seq(x.unsqueeze(1) * quantile_embed)