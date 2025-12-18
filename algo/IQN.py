import torch as th
from utils.network import IQN_net
import torch.optim as optim
import tqdm
import numpy as np
from utils.func import iqn_loss, adjust

class IQN:
    def __init__(self, env, device):
        self.env = env
        self.epoch = 300
        self.gamma = 0.99
        self.device = device
        self.N = 8
        self.N_hat = 8
        self.IQN_network = IQN_net(input_dim=3).to(self.device)
        self.optimizer = optim.Adam(self.IQN_network.parameters(), lr=0.01)
        self.best_reward = -300

    def save(self):
        th.save(self.IQN_network.state_dict(), "./result/model/IQN_net.pth")

    def load(self):
        self.IQN_network.load_state_dict(th.load("./result/model/IQN_net.pth", map_location=self.device))

    def train(self):
        self.IQN_network.train()
        total_reward_list = []
        for i in tqdm.tqdm(range(self.epoch)):
            observation, _ = self.env.reset()
            episode_over = False
            max_x = min_x = observation[0]
            while not episode_over:
                tau = th.rand((1, self.N)).to(self.device)
                tau_hat = th.rand((1, self.N_hat)).to(self.device)

                # choice action (epsilon greedy)
                pred_q = []
                for a in range(3):
                    x = th.from_numpy(np.concatenate([observation, np.array([a], dtype=np.float32)])).unsqueeze(0).to(
                        self.device)
                    with th.no_grad():
                        pred_quantile = self.IQN_network(x, tau).squeeze()
                    assert pred_quantile.shape == (self.N, )
                    pred_q.append(pred_quantile.mean())
                action = th.argmax(th.tensor(pred_q)).item()
                if th.rand(1).item() < 0.1:
                    action = np.random.randint(0, 3)

                # step
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                reward, max_x, min_x = adjust(next_observation, reward, max_x, min_x)

                # update
                # Bellman Optimal Function
                pred_q = []
                pred_quantile_list = []
                for a in range(3):
                    x = th.from_numpy(np.concatenate([next_observation, np.array([a], dtype=np.float32)])).unsqueeze(
                        0).to(self.device)
                    with th.no_grad():
                        next_quantile = self.IQN_network(x, tau_hat).squeeze()
                    assert next_quantile.shape == (self.N_hat, )
                    pred_q.append(next_quantile.mean())
                    pred_quantile_list.append(next_quantile.unsqueeze(0))
                idx = th.argmax(th.tensor(pred_q)).item()
                target_quantile = reward + self.gamma * pred_quantile_list[idx] * (1 - terminated)

                cur_pair = th.from_numpy(np.concatenate([observation, np.array([action], dtype=np.float32)])).unsqueeze(
                    0).to(self.device)
                pred_quantile = self.IQN_network(cur_pair, tau).squeeze(-1)

                loss = iqn_loss(pred_quantile, target_quantile.detach(), tau)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                observation = next_observation
                episode_over = terminated or truncated

            if (i + 1) % 10 == 0:
                total_reward = self.test()
                if total_reward > self.best_reward:
                    self.best_reward = total_reward
                    self.save()
                total_reward_list.append(total_reward)

        return total_reward_list

    def test(self):
        self.IQN_network.eval()
        observation, _ = self.env.reset()
        episode_over = False
        total_reward = 0
        while not episode_over:
            tau = th.rand((1, self.N)).to(self.device)

            # choice action (epsilon greedy)
            pred_q = []
            for a in range(3):
                x = th.from_numpy(np.concatenate([observation, np.array([a], dtype=np.float32)])).unsqueeze(0).to(
                    self.device)
                with th.no_grad():
                    pred_quantile = self.IQN_network(x, tau).squeeze()
                pred_q.append(pred_quantile.mean())
            action = th.argmax(th.tensor(pred_q)).item()

            # step
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            observation = next_observation
            episode_over = terminated or truncated
            total_reward -= 1
        return total_reward