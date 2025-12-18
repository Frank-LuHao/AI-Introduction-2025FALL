import torch as th
from utils.network import QR_DQN_net
import torch.optim as optim
import tqdm
import numpy as np
from utils.func import qtd_loss, adjust

class QRDQN:
    def __init__(self, env, device):
        self.env = env
        self.epoch = 300
        self.gamma = 0.99
        self.device = device
        self.tau = th.tensor([i / 32 for i in range(1, 32, 2)], device=self.device)
        self.QR_DQN_network = QR_DQN_net(input_dim=3, output_dim=16).to(self.device)
        self.optimizer = optim.Adam(self.QR_DQN_network.parameters(), lr=0.01)
        self.best_reward = -300

    def save(self):
        th.save(self.QR_DQN_network.state_dict(), "./result/model/QR_DQN_net.pth")

    def load(self):
        self.QR_DQN_network.load_state_dict(th.load("./result/model/QR_DQN_net.pth", map_location=self.device))

    def train(self):
        self.QR_DQN_network.train()
        total_reward_list = []
        for i in tqdm.tqdm(range(self.epoch)):
            observation, _ = self.env.reset()
            episode_over = False
            max_x = min_x = observation[0]
            while not episode_over:
                # choice action (epsilon greedy)
                pred_q = []
                for a in range(3):
                    x = th.from_numpy(np.concatenate([observation, np.array([a], dtype=np.float32)])).unsqueeze(0).to(
                        self.device)
                    with th.no_grad():
                        pred_quantile = self.QR_DQN_network(x).squeeze()
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
                        next_quantile = self.QR_DQN_network(x)
                    pred_q.append(next_quantile.squeeze().mean())
                    pred_quantile_list.append(next_quantile)
                idx = th.argmax(th.tensor(pred_q)).item()
                target_quantile = reward + self.gamma * pred_quantile_list[idx] * (1 - terminated)

                cur_pair = th.from_numpy(np.concatenate([observation, np.array([action], dtype=np.float32)])).unsqueeze(
                    0).to(self.device)
                pred_quantile = self.QR_DQN_network(cur_pair)

                loss = qtd_loss(pred_quantile, target_quantile.detach(), self.tau)
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
        self.QR_DQN_network.eval()
        observation, _ = self.env.reset()
        episode_over = False
        total_reward = 0
        while not episode_over:
            # choice action (epsilon greedy)
            pred_q = []
            for a in range(3):
                x = th.from_numpy(np.concatenate([observation, np.array([a], dtype=np.float32)])).unsqueeze(0).to(
                    self.device)
                with th.no_grad():
                    pred_quantile = self.QR_DQN_network(x).squeeze()
                pred_q.append(pred_quantile.mean())
            action = th.argmax(th.tensor(pred_q)).item()

            # step
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            observation = next_observation
            episode_over = terminated or truncated
            total_reward -= 1
        return total_reward