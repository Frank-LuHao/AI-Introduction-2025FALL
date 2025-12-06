from network import QR_DQN
import torch as th
import torch.optim as optim
from func import qtd_loss
import tqdm
import matplotlib.pyplot as plt
import scipy.stats as stats

def train(algo):
    if algo == 'QR_DQN':
        QR_DQN_network = QR_DQN()
        optimizer = optim.Adam(QR_DQN_network.parameters(), lr=0.001)
        tau = th.tensor([i/32 for i in range(1, 32, 2)])
        batch_size = 64

        QR_DQN_network.train()
        for _ in tqdm.tqdm(range(10000)):
            x = th.ones((batch_size, 16,), dtype=th.float)

            # uniform test
            # reward = th.rand((batch_size, 1)).expand(-1, 16)

            # reward = th.tensor([[0.0312, 0.0938, 0.1562, 0.2188, 0.2812, 0.3438, 0.4062, 0.4688, 0.5312,
            # 0.5938, 0.6562, 0.7188, 0.7812, 0.8438, 0.9062, 0.9688]]).expand(batch_size, 16)

            # 1&2 test
            # if th.rand((1)).item() < 0.5:
            #     reward = th.ones((1,)).expand(1, 16) + 1
            # else:
            #     reward = th.ones((1,)).expand(1, 16)

            # normal test
            reward = th.randn((batch_size, 1)).expand(-1, 16)

            pred_quantile = QR_DQN_network(x)
            loss = qtd_loss(pred_quantile, reward, tau)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        QR_DQN_network.eval()
        x = th.ones((1, 16,), dtype=th.float)
        pred_quantile = QR_DQN_network(x).squeeze()
        print(pred_quantile)
        # 计算正态分布的分位数
        normal_quantiles = stats.norm.ppf(tau.numpy())
        print(f'tau: {normal_quantiles}')

        plt.plot(pred_quantile.detach().numpy(), tau.numpy())
        plt.plot(normal_quantiles, tau.numpy(), linestyle='--')
        plt.xlabel('tau')
        plt.show()

    if algo == 'C51':
        C51_network = C51()
        optimizer = optim.Adam(C51_network.parameters(), lr=0.01)
        batch_size = 32