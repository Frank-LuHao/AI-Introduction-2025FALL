import numpy as np
import gymnasium
import os
from algo import QRDQN, C51, IQN
import matplotlib.pyplot as plt

class experiment2:
    def __init__(self, device, algo):
        # https://gymnasium.farama.org/environments/classic_control/mountain_car/
        self.env = gymnasium.make("MountainCar-v0")
        self.device = device
        self.algo_name = algo
        self.Algo = None

    def set_Env(self, render=False):
        if not render:
            self.env = gymnasium.make("MountainCar-v0")
        else:
            self.env = gymnasium.make("MountainCar-v0", render_mode="rgb_array")
            self.env = gymnasium.wrappers.RecordVideo(self.env, f"./result/video/{self.algo_name}")

    def set_Algo(self):
        if self.algo_name == "QR_DQN":
            self.Algo = QRDQN(self.env, self.device)
        elif self.algo_name == "C51":
            self.Algo = C51(self.env, self.device)
        else:
            self.Algo = IQN(self.env, self.device)

    def train(self, times):
        self.set_Env()
        all_rewards = []
        for _ in range(times):
            self.set_Algo()
            reward_list = self.Algo.train()
            all_rewards.append(reward_list)
        all_rewards = np.array(all_rewards)
        np.save(f"./result/reward/{self.algo_name}_rewards", all_rewards)

    def test(self):
        self.set_Env(True)
        self.Algo.load()
        reward = self.Algo.test()
        print(f'reward:{reward}')

    # def vis(self):
    #     data = np.load("./result/reward/" + self.algo_name + "_rewards.npy")
    #     mean_reward = np.mean(data, axis=0)
    #     std_reward = np.std(data, axis=0)

    #     window_size = 10
    #     window = np.ones(window_size) / window_size
    #     mean_smooth = np.convolve(mean_reward, window, mode="valid")
    #     trim_start = window_size - 1

    #     x = np.arange(len(mean_reward))[trim_start:]
    #     std_smooth = std_reward[trim_start:]
    #     p = plt.plot(x, mean_smooth, label="mean")
    #     color = p[0].get_color()
    #     plt.fill_between(x, np.maximum(mean_smooth - std_smooth, -200), mean_smooth + std_smooth, color=color, alpha=0.2)
    #     plt.grid(True)
    #     plt.show()

    def vis(self):
        reward_dir = "./result/reward/"
        npy_files = ['IQN_rewards.npy', 'QR_DQN_rewards.npy', 'C51_rewards.npy']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
        window_size = 3
        window = np.ones(window_size) / window_size
        for idx, npy_file in enumerate(npy_files):
            data = np.load(os.path.join(reward_dir, npy_file))
            mean_reward = np.mean(data, axis=0)
            std_reward = np.std(data, axis=0)
            # 移动平均
            mean_smooth = np.convolve(mean_reward, window, mode="valid")
            std_smooth = np.convolve(std_reward, window, mode="valid")
            algo_name = npy_file.replace("_rewards.npy", "")
            x = np.arange(len(mean_smooth))
            plt.plot(x, mean_smooth, label=algo_name, color=colors[idx % len(colors)])
            plt.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth, color=colors[idx % len(colors)], alpha=0.2)
            if npy_file == 'C51_rewards.npy':
                print(mean_smooth)

        plt.xlabel("Episode")
        plt.ylabel("Mean Reward (Moving Avg)")
        plt.legend()
        plt.grid(True)
        plt.show()