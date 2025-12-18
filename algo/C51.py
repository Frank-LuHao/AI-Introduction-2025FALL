import torch as th
from utils.network import C51_net
import torch.optim as optim
import tqdm
import numpy as np
from utils.func import c51_loss, adjust

class C51:
    def __init__(self, env, device):
        self.env = env
        self.epoch = 300
        self.gamma = 0.99
        self.device = device
        self.v_min = th.tensor(-200.0).to(self.device)
        self.v_max = th.tensor(0.0).to(self.device)
        self.atom_num = 51
        self.atoms = th.linspace(self.v_min, self.v_max, self.atom_num, dtype=th.float).to(self.device)
        self.C51_network = C51_net(input_dim=3, atom_num=self.atom_num).to(self.device)
        self.optimizer = optim.Adam(self.C51_network.parameters(), lr=0.01)
        self.best_reward = -300

    def save(self):
        th.save(self.C51_network.state_dict(), "./result/model/C51_net.pth")

    def load(self):
        self.C51_network.load_state_dict(th.load("./result/model/C51_net.pth", map_location=self.device))

    def train(self):
        self.C51_network.train()
        total_reward_list = []
        for i in tqdm.tqdm(range(self.epoch)):
            observation, _ = self.env.reset()
            episode_over = False
            max_x = min_x = observation[0]
            while not episode_over:
                # choice action (epsilon greedy)
                pred_q = []
                for a in range(3):
                    x = th.from_numpy(np.concatenate([observation, np.array([a], dtype=np.float32)])).unsqueeze(0).to(self.device)
                    with th.no_grad():
                        pred_pd = self.C51_network(x).squeeze()
                    pred_q.append((pred_pd * self.atoms).sum())
                action = th.argmax(th.tensor(pred_q)).item()
                if th.rand(1).item() < 0.1:
                    action = np.random.randint(0, 3)

                # step
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                reward, max_x, min_x = adjust(next_observation, reward, max_x, min_x)

                # update
                # Bellman Optimal Function
                pred_q = []
                pred_distribution_list = []
                for a in range(3):
                    x = th.from_numpy(np.concatenate([next_observation, np.array([a], dtype=np.float32)])).unsqueeze(0).to(self.device)
                    with th.no_grad():
                        next_pd = self.C51_network(x).squeeze()
                    assert next_pd.shape == (self.atom_num, )
                    pred_q.append((next_pd * self.atoms).sum())
                    pred_distribution_list.append(next_pd)
                idx = th.argmax(th.tensor(pred_q)).item()
                # 投影转换
                delta = (self.v_max - self.v_min) / (self.atom_num - 1)
                trans_atoms = th.clamp(reward + (1 - terminated) * self.gamma * self.atoms, self.v_min, self.v_max)
                assert trans_atoms.shape == (self.atom_num, )
                distance = 1 - th.abs(trans_atoms.unsqueeze(0).expand(self.atom_num, -1) - self.atoms.unsqueeze(-1)) / delta
                distance = th.clamp(distance, th.tensor(0.0).to(self.device), th.tensor(1.0).to(self.device))
                assert distance.shape == (self.atom_num, self.atom_num)
                target_distribution = (distance @ pred_distribution_list[idx].view(self.atom_num, 1)).view(1, self.atom_num)
                assert target_distribution.shape == (1, self.atom_num)

                cur_pair = th.from_numpy(np.concatenate([observation, np.array([action], dtype=np.float32)])).unsqueeze(0).to(self.device)
                pred_distribution = self.C51_network(cur_pair)

                # print(f'pred_distribution: {pred_distribution.shape}\n{pred_distribution}')
                # print(f'target_distribution: {target_distribution.shape}\n{target_distribution}')
                loss = c51_loss(pred_distribution, target_distribution.detach())
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
        self.C51_network.eval()
        observation, _ = self.env.reset()
        episode_over = False
        total_reward = 0
        while not episode_over:
            # choice action (epsilon greedy)
            pred_q = []
            for a in range(3):
                x = th.from_numpy(np.concatenate([observation, np.array([a], dtype=np.float32)])).unsqueeze(0).to(self.device)
                with th.no_grad():
                    pred_pd = self.C51_network(x).squeeze()
                pred_q.append((pred_pd * self.atoms).sum())
            action = th.argmax(th.tensor(pred_q)).item()

            # step
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            observation = next_observation
            episode_over = terminated or truncated
            total_reward -= 1
        return total_reward