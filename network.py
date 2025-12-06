import torch.nn as nn
import torch.nn.functional as F

# QR_DQN 算法
class QR_DQN(nn.Module):
    def __init__(self):
        super(QR_DQN, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )

    def forward(self, x):
        return self.seq(x)

# C51 算法
class C51(nn.Module):
    def __init__(self, atom_num):
        super(C51, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, atom_num)
        )
    def forward(self, x):
        return F.softmax(self.seq(x), dim=-1)