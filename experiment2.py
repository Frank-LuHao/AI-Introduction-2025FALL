import numpy as np
import gymnasium
from algo import QRDQN
import matplotlib.pyplot as plt

class experiment2:
    def __init__(self, device):
        # https://gymnasium.farama.org/environments/classic_control/mountain_car/
        self.env = gymnasium.make("MountainCar-v0")
        self.device = device
        self.algo_name = None
        self.Algo = None

    def set_Env(self, render=False):
        if not render:
            self.env = gymnasium.make("MountainCar-v0")
        else:
            self.env = gymnasium.make("MountainCar-v0", render_mode="human")

    def set_Algo(self, algo):
        self.algo_name = algo
        if algo == "QR_DQN":
            self.Algo = QRDQN(self.env, self.device)
        elif algo == "C51":
            self.Algo = QRDQN(self.env, self.device)
        else:
            self.Algo = QRDQN(self.env, self.device)

    def train(self, times):
        self.set_Env()
        all_rewards = []
        for _ in range(times):
            reward_list = self.Algo.train()
            all_rewards.append(reward_list)
        all_rewards = np.array(all_rewards)
        np.save(f"./result/reward/{self.algo_name}_rewards", all_rewards)

    def test(self):
        self.set_Env(True)
        self.Algo.load()
        self.Algo.test()

    def vis(self):
        data = np.load("./result/reward/" + self.algo_name + "_rewards.npy")
        mean_reward = np.mean(data, axis=0)
        std_reward = np.std(data, axis=0)

        window_size = 10
        window = np.ones(window_size) / window_size
        mean_smooth = np.convolve(mean_reward, window, mode="valid")
        trim_start = window_size - 1

        x = np.arange(len(mean_reward))[trim_start:]
        std_smooth = std_reward[trim_start:]
        p = plt.plot(x, mean_smooth, label="mean")
        color = p[0].get_color()
        plt.fill_between(x, np.maximum(mean_smooth - std_smooth, -200), mean_smooth + std_smooth, color=color, alpha=0.2)
        plt.grid(True)
        plt.show()


            # if algo == 'C51':
        #     v_min = -5
        #     v_max = 5
        #     atom_num = 16
        #     batch_size = 64
        #     atoms = th.linspace(v_min, v_max, atom_num)
        #     C51_network = C51_net(atom_num)
        #     optimizer = optim.Adam(C51_network.parameters(), lr=0.001)
        #
        #     C51_network.train()
        #     for i in tqdm.tqdm(range(10000)):
        #         x = th.ones((batch_size, 16,), dtype=th.float)
        #
        #         # same value test
        #         # reward = th.zeros((1, 16))
        #         # reward[0][3] = 1
        #         # reward = reward.expand(batch_size, 16)
        #
        #         # same value test2
        #         # reward = th.tensor([[0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625,
        #         # 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]]).expand(batch_size, 16)
        #
        #         # uniform test
        #         reward = th.zeros((batch_size, atom_num))
        #         for j in range(batch_size):
        #             idx = th.randint(0, atom_num, (1,)).item()
        #             reward[j][idx] = 1
        #
        #
        #         # 投影转换
        #         # 由于这里 reward 直接就是 target distribution, 所以省略这个步骤
        #
        #         pred_distribution = C51_network(x)
        #         loss = c51_loss(pred_distribution, reward)
        #         # print(f"i : {loss}")
        #
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #
        #     C51_network.eval()
        #     x = th.ones((1, 16,), dtype=th.float)
        #     pred_distribution = C51_network(x).squeeze()
        #     print(pred_distribution)
        #
        # if algo == "IQN":
        #     batch_size = 32
        #     N = 8
        #     N_hat = 8
        #     IQN_network = IQN_net()
        #     optimizer = optim.Adam(IQN_network.parameters(), lr=0.001)
        #
        #     IQN_network.train()
        #     for _ in tqdm.tqdm(range(10000)):
        #         x = th.ones((batch_size, 16,), dtype=th.float)
        #         tau = th.rand((batch_size, N))
        #
        #         # 单点分布
        #         # reward = th.zeros((batch_size, N_hat)) + 1
        #
        #         # uniform 分布
        #         # reward = th.rand(batch_size, N_hat)
        #
        #         # Guass 分布
        #         reward = th.randn(batch_size, N_hat)
        #
        #         pred_value = IQN_network(x, tau)
        #         assert pred_value.shape == (batch_size, N, 1)
        #         loss = iqn_loss(pred_value.squeeze(2), reward, tau)
        #
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #
        #     IQN_network.eval()
        #     x = th.ones((1, 16,), dtype=th.float)
        #     tau = th.linspace(0, 1, 10).unsqueeze(0)
        #     # print(tau)
        #     # 计算正态分布的分位数
        #     normal_quantiles = stats.norm.ppf(tau.squeeze().numpy())
        #     print(f'tau: {normal_quantiles}')
        #     pred_value = IQN_network(x, tau)
        #     print(pred_value)