from network import QR_DQN, C51
import torch as th
import torch.optim as optim
from func import qtd_loss, c51_loss
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

            # same value test
            # reward = th.zeros((batch_size, 16)) + 1

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
        v_max = -5
        v_min = 5
        atom_num = 16
        batch_size = 64
        atoms = th.linspace(v_min, v_max, atom_num)
        C51_network = C51(atom_num)
        optimizer = optim.Adam(C51_network.parameters(), lr=0.001)

        C51_network.train()
        for i in tqdm.tqdm(range(10000)):
            x = th.ones((batch_size, 16,), dtype=th.float)

            # same value test
            # reward = th.zeros((1, 16))
            # reward[0][3] = 1
            # reward = reward.expand(batch_size, 16)

            # same value test2
            # reward = th.tensor([[0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625,
            # 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]]).expand(batch_size, 16)

            # uniform test
            reward = th.zeros((batch_size, atom_num))
            for i in range(batch_size):
                idx = th.randint(0, atom_num, (1,)).item()
                reward[i][idx] = 1

            # todo: normal test

            # 投影转换
            # 由于这里 reward 直接就是 target distribution, 所以省略这个步骤

            pred_distribution = C51_network(x)
            loss = c51_loss(pred_distribution, reward)
            # print(f"i : {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        C51_network.eval()
        x = th.ones((1, 16,), dtype=th.float)
        pred_distribution = C51_network(x).squeeze()
        print(pred_distribution)