import numpy as np
import gymnasium
from algo import QRDQN, C51, IQN
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
            self.Algo = C51(self.env, self.device)
        else:
            self.Algo = IQN(self.env, self.device)

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