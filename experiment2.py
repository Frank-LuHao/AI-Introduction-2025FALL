import numpy as np
from network import QR_DQN, C51, IQN
import torch as th
import torch.optim as optim
from func import qtd_loss, c51_loss, iqn_loss
import tqdm
import matplotlib.pyplot as plt
import scipy.stats as stats
import gymnasium

class experiment2:
    def __init__(self, device):
        # https://gymnasium.farama.org/environments/classic_control/mountain_car/
        self.env = gymnasium.make("MountainCar-v0")
        self.device = device

    def train(self, algo):
        if algo == 'QR_DQN':
            epochs = 300
            gamma = 1
            tau = th.tensor([i / 32 for i in range(1, 32, 2)], device=self.device)
            QR_DQN_network = QR_DQN(input_dim=3, output_dim=16).to(self.device)
            optimizer = optim.Adam(QR_DQN_network.parameters(), lr=0.01)

            QR_DQN_network.train()
            total_reward_list = []
            for _ in tqdm.tqdm(range(epochs)):
                observation, _ = self.env.reset()
                episode_over = False
                total_reward = 0
                while not episode_over:
                    # choice action (epsilon greedy)
                    pred_q = []
                    for a in range(3):
                        x = th.from_numpy(np.concatenate([observation, np.array([a], dtype=np.float32)])).unsqueeze(0).to(self.device)
                        with th.no_grad():
                            pred_quantile = QR_DQN_network(x).squeeze()
                        pred_q.append(pred_quantile.mean())
                    action = th.argmax(th.tensor(pred_q)).item()
                    if th.rand(1).item() < 0.3:
                        action = np.random.randint(0, 3)

                    # step
                    next_observation, reward, terminated, truncated, _ = self.env.step(action)

                    # update
                    # Bellman Optimal Function
                    pred_q = []
                    pred_quantile_list = []
                    for a in range(3):
                        x = th.from_numpy(np.concatenate([next_observation, np.array([a], dtype=np.float32)])).unsqueeze(0).to(self.device)
                        with th.no_grad():
                            next_quantile = QR_DQN_network(x)
                        pred_q.append(next_quantile.squeeze().mean())
                        pred_quantile_list.append(next_quantile)
                    idx = th.argmax(th.tensor(pred_q)).item()
                    target_quantile = reward + gamma * pred_quantile_list[idx] * (1 - terminated)

                    cur_pair = th.from_numpy(np.concatenate([observation, np.array([action], dtype=np.float32)])).unsqueeze(0).to(self.device)
                    pred_quantile = QR_DQN_network(cur_pair)

                    loss = qtd_loss(pred_quantile, target_quantile.detach(), tau)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_reward += reward
                    observation = next_observation
                    episode_over = terminated or truncated
                total_reward_list.append(total_reward)

            plt.plot(total_reward_list)
            plt.show()

            # Test
            self.env = gymnasium.make("MountainCar-v0", render_mode="human")
            QR_DQN_network.eval()
            observation, _ = self.env.reset()
            episode_over = False
            while not episode_over:
                # choice action (epsilon greedy)
                pred_q = []
                for a in range(3):
                    x = th.from_numpy(np.concatenate([observation, np.array([a], dtype=np.float32)])).unsqueeze(0).to(self.device)
                    with th.no_grad():
                        pred_quantile = QR_DQN_network(x).squeeze()
                    pred_q.append(pred_quantile.mean())
                action = th.argmax(th.tensor(pred_q)).item()
                if th.rand(1).item() < 0.3:
                    action = np.random.randint(0, 3)

                # step
                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                observation = next_observation
                episode_over = terminated or truncated

        if algo == 'C51':
            v_min = -5
            v_max = 5
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
                for j in range(batch_size):
                    idx = th.randint(0, atom_num, (1,)).item()
                    reward[j][idx] = 1

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

        if algo == "IQN":
            batch_size = 32
            N = 8
            N_hat = 8
            IQN_network = IQN()
            optimizer = optim.Adam(IQN_network.parameters(), lr=0.001)

            IQN_network.train()
            for _ in tqdm.tqdm(range(10000)):
                x = th.ones((batch_size, 16,), dtype=th.float)
                tau = th.rand((batch_size, N))

                # 单点分布
                # reward = th.zeros((batch_size, N_hat)) + 1

                # uniform 分布
                # reward = th.rand(batch_size, N_hat)

                # Guass 分布
                reward = th.randn(batch_size, N_hat)

                pred_value = IQN_network(x, tau)
                assert pred_value.shape == (batch_size, N, 1)
                loss = iqn_loss(pred_value.squeeze(2), reward, tau)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            IQN_network.eval()
            x = th.ones((1, 16,), dtype=th.float)
            tau = th.linspace(0, 1, 10).unsqueeze(0)
            # print(tau)
            # 计算正态分布的分位数
            normal_quantiles = stats.norm.ppf(tau.squeeze().numpy())
            print(f'tau: {normal_quantiles}')
            pred_value = IQN_network(x, tau)
            print(pred_value)