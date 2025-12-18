from utils.network import QR_DQN_net, C51_net, IQN_net
import torch as th
import torch.optim as optim
from utils.func import qtd_loss, c51_loss, iqn_loss
import tqdm
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

class experiment1:
    def train(self, algo):
        if algo == 'QR_DQN':
            QR_DQN_network = QR_DQN_net(input_dim=16, output_dim=16)
            optimizer = optim.Adam(QR_DQN_network.parameters(), lr=0.001)
            tau = th.tensor([i/32 for i in range(1, 32, 2)])
            batch_size = 64

            QR_DQN_network.train()
            for _ in tqdm.tqdm(range(1000)):
                x = th.ones((batch_size, 16,), dtype=th.float)

                # same value test
                # reward = th.zeros((batch_size, 16)) + 1

                # uniform test
                # reward = th.rand((batch_size, 1)).expand(-1, 16)

                # 1&2 test
                # if th.rand((1)).item() < 0.5:
                #     reward = th.ones((1,)).expand(1, 16) + 1
                # else:
                #     reward = th.ones((1,)).expand(1, 16)

                # normal test
                # reward = th.randn((batch_size, 1)).expand(-1, 16)

                # Guass Mixture test
                mask = (th.rand(batch_size, 1) < 0.3).float()
                mu = mask * -2 + (1 - mask) * 3
                sigma = mask * 0.5 + (1 - mask) * 1.0
                reward = th.normal(mu, sigma).expand(-1, 16)

                pred_quantile = QR_DQN_network(x)
                loss = qtd_loss(pred_quantile, reward, tau)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            QR_DQN_network.eval()
            x = th.ones((1, 16,), dtype=th.float)
            pred_quantile = QR_DQN_network(x).squeeze()
            print(pred_quantile)

            x_dense = th.linspace(0, 1, 500)
            # 计算正态分布的分位数
            # normal_quantiles = stats.norm.ppf(x_dense)
            # print(f'tau: {normal_quantiles}')

            # 计算高斯混合分布的分位数
            normal_quantiles = []
            for t in x_dense:
                def mixture_cdf(x):
                    return 0.3 * stats.norm.cdf(x, loc=-2, scale=0.5) + 0.7 * stats.norm.cdf(x, loc=3, scale=1.0)
                # 使用二分法寻找分位数
                low, high = -10, 10
                for _ in range(100):
                    mid = (low + high) / 2
                    if mixture_cdf(mid) < t:
                        low = mid
                    else:
                        high = mid
                normal_quantiles.append((low + high) / 2)
            normal_quantiles = th.tensor(normal_quantiles, dtype=th.float)
            print(f'tau: {normal_quantiles.numpy()}')

            plt.plot(pred_quantile.detach().numpy(), tau.numpy())
            plt.plot(normal_quantiles, x_dense, linestyle='--')
            plt.xlabel('tau')
            plt.show()

        if algo == 'C51':
            v_min = -3
            v_max = 3
            atom_num = 31
            batch_size = 64
            atoms = th.linspace(v_min, v_max, atom_num)
            C51_network = C51_net(16, atom_num)
            optimizer = optim.Adam(C51_network.parameters(), lr=0.0001)

            C51_network.train()
            for _ in tqdm.tqdm(range(10000)):
                x = th.ones((batch_size, 16,), dtype=th.float)

                # same value test
                # reward = th.zeros((1, 16))
                # reward[0][3] = 1
                # reward = reward.expand(batch_size, 16)

                # same value test2
                # reward = th.tensor([[0.00443185, 0.00630672, 0.00884645, 0.01223152, 0.0166701,  0.02239453,
                #                      0.02965459, 0.03870685, 0.04980009, 0.06315655, 0.07895015, 0.09728226,
                #                      0.11815728, 0.14145995, 0.16693703, 0.19418604, 0.22265349, 0.25164433,
                #                      0.28034377, 0.30785125, 0.33322457, 0.35553253, 0.37391059, 0.38761661,
                #                      0.39608021, 0.39894228, 0.39608021, 0.38761661, 0.37391059, 0.35553253,
                #                      0.33322457, 0.30785125, 0.28034377, 0.25164433, 0.22265349, 0.19418604,
                #                      0.16693703, 0.14145995, 0.11815728, 0.09728226, 0.07895015, 0.06315655,
                #                      0.04980009, 0.03870685, 0.02965459, 0.02239453, 0.0166701, 0.01223152,
                #                      0.00884645, 0.00630672, 0.00443185]]).expand(batch_size, atom_num)

                # uniform test
                # reward = th.zeros((batch_size, atom_num))
                # for j in range(batch_size):
                #     idx = th.randint(0, atom_num, (1,)).item()
                #     reward[j][idx] = 1

                # normal test - 通过采样统计构建离散概率分布
                num_samples = 1000
                mu = th.randn(1).item()
                sigma = th.rand(1).item() + 0.1
                # 生成样本
                samples = th.normal(mu, sigma, size=(num_samples,))
                # 统计每个 atom 区间内的样本数量
                delta_z = (v_max - v_min) / (atom_num - 1)
                prob_dist = th.zeros(atom_num)
                for i in range(atom_num):
                    # 计算区间边界 [atom - delta/2, atom + delta/2)
                    lower = atoms[i] - delta_z / 2
                    upper = atoms[i] + delta_z / 2
                    # 统计落在该区间内的样本数量
                    count = ((samples >= lower) & (samples < upper)).sum().item()
                    prob_dist[i] = count / num_samples
                # 确保概率和为 1 (处理边界情况)
                prob_dist /= prob_dist.sum()
                reward = prob_dist.unsqueeze(0).expand(batch_size, atom_num)

                # Gaussian Mixture test - 通过采样统计构建离散概率分布
                # num_samples = 1000
                # mix_prob = 0.3  # 混合概率
                # # 高斯混合参数
                # mu1, sigma1 = -2, 0.5
                # mu2, sigma2 = 3, 1.0
                # # 根据混合概率采样
                # mask = (th.rand(num_samples) < mix_prob).float()
                # samples = mask * th.normal(mu1, sigma1, size=(num_samples,)) + \
                #           (1 - mask) * th.normal(mu2, sigma2, size=(num_samples,))
                # # 统计每个 atom 区间内的样本数量
                # delta_z = (v_max - v_min) / (atom_num - 1)
                # prob_dist = th.zeros(atom_num)
                # for i in range(atom_num):
                #     lower = atoms[i] - delta_z / 2
                #     upper = atoms[i] + delta_z / 2
                #     count = ((samples >= lower) & (samples < upper)).sum().item()
                #     prob_dist[i] = count / num_samples
                # # 确保概率和为 1
                # prob_dist /= prob_dist.sum()
                # reward = prob_dist.unsqueeze(0).expand(batch_size, atom_num)

                pred_distribution = C51_network(x)
                loss = c51_loss(pred_distribution, reward)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            C51_network.eval()
            x = th.ones((1, 16,), dtype=th.float)
            pred_distribution = C51_network(x).squeeze()
            print(pred_distribution)


            # 作概率密度图可视化
            delta_z = (v_max - v_min) / (atom_num - 1)
            pred_distribution_density = pred_distribution / delta_z

            # 绘制离散预测分布 - 使用柱状图
            plt.bar(atoms.numpy(), pred_distribution_density.detach().numpy(),
                    width=delta_z * 0.8, alpha=0.6, label='Predicted')

            # 绘制理论混合高斯分布 - 使用密集采样点
            # x_dense = np.linspace(v_min, v_max, 500)  # 500个点用于平滑曲线
            # pdf1 = stats.norm.pdf(x_dense, loc=-2, scale=0.5)
            # pdf2 = stats.norm.pdf(x_dense, loc=3, scale=1.0)
            # pdf = 0.3 * pdf1 + 0.7 * pdf2

            # 计算正态分布的概率密度函数
            x_dense = np.linspace(v_min, v_max, 500)
            pdf = stats.norm.pdf(x_dense, loc=0, scale=1.0)

            plt.plot(x_dense, pdf, linestyle='--', color='red', linewidth=2, label='True Distribution')

            plt.xlabel('Value')
            plt.ylabel('Probability Density')
            plt.legend()
            plt.show()

        if algo == "IQN":
            batch_size = 32
            N = 8
            N_hat = 8
            IQN_network = IQN_net(16)
            optimizer = optim.Adam(IQN_network.parameters(), lr=0.001)

            IQN_network.train()
            for _ in tqdm.tqdm(range(10000)):
                x = th.ones((batch_size, 16,), dtype=th.float)
                tau = th.rand((batch_size, N))

                # 单点分布
                # reward = th.zeros((batch_size, N_hat)) + 1

                # uniform 分布
                # reward = th.rand(batch_size, N_hat)

                # Normal test
                # reward = th.randn(batch_size, N_hat)

                # Guass Mixture test
                mask = (th.rand(batch_size, 1) < 0.3).float()
                mu = mask * -2 + (1 - mask) * 3
                sigma = mask * 0.5 + (1 - mask) * 1.0
                reward = th.normal(mu, sigma).expand(-1, N_hat)

                pred_value = IQN_network(x, tau)
                assert pred_value.shape == (batch_size, N, 1)
                loss = iqn_loss(pred_value.squeeze(2), reward, tau)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            IQN_network.eval()
            x = th.ones((1, 16,), dtype=th.float)
            tau = th.linspace(0, 1, 500).unsqueeze(0)
            pred_value = IQN_network(x, tau)
            print(pred_value)

            x_dense = np.linspace(0, 1, 500)
            # 计算正态分布的分位数
            # normal_quantiles = stats.norm.ppf(x_dense)
            # print(f'tau: {normal_quantiles}')

            # 计算高斯混合分布的分位数
            normal_quantiles = []
            for t in x_dense:
                def mixture_cdf(x):
                    return 0.3 * stats.norm.cdf(x, loc=-2, scale=0.5) + 0.7 * stats.norm.cdf(x, loc=3, scale=1.0)
                # 使用二分法寻找分位数
                low, high = -10, 10
                for _ in range(100):
                    mid = (low + high) / 2
                    if mixture_cdf(mid) < t:
                        low = mid
                    else:
                        high = mid
                normal_quantiles.append((low + high) / 2)
            normal_quantiles = th.tensor(normal_quantiles, dtype=th.float)
            print(f'tau: {normal_quantiles.numpy()}')

            plt.plot(pred_value.squeeze().detach().numpy(), tau.squeeze().numpy())
            plt.plot(normal_quantiles, x_dense, linestyle='--')
            plt.xlabel('tau')
            plt.show()