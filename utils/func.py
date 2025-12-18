import torch as th

# reward 重构
def adjust(next_obs, reward, min_x, max_x):
    x = next_obs[0]
    speed = next_obs[1]
    if x > max_x:
        reward += (x - max_x) * 10
        max_x = x
    elif x < min_x:
        reward += (min_x - x) * 10
        min_x = x
    if abs(speed) > 0.01:
        reward += abs(speed) * 5
    else:
        reward -= 1

    return reward, max_x, min_x

def c51_loss(predicted_distribution, target_distribution):
    return -th.mean(target_distribution * th.log(predicted_distribution + 1e-10))
    # return F.cross_entropy(target_distribution, predicted_distribution, reduction='mean') # th 中的 cross_entropy 函数好像和理解中的不太一样 ？
    # return F.mse_loss(target_distribution, predicted_distribution, reduction='mean')  # 既然概率都基于相同的atoms上，那为什么不使用mse_loss ?

def qtd_loss(predicted_quantile, target_quantile, tau):
    shape = predicted_quantile.shape
    x = target_quantile.unsqueeze(1) - predicted_quantile.unsqueeze(2)
    assert x.shape == (shape[0], shape[1], shape[1])
    # huber = F.huber_loss(x, th.zeros_like(x), reduction='none')  # 这里使用huber loss的分布有偏，思考
    weight = tau.unsqueeze(0).unsqueeze(2) - x.le(0).float()
    assert weight.shape == (shape[0], shape[1], shape[1])
    loss = (weight * x).mean(dim=(0,)).sum(dim=(0, 1))
    return loss

def iqn_loss(predicted_quantile, target_quantile, tau):
    batch_size = predicted_quantile.shape[0]
    size1 = predicted_quantile.shape[1]
    size2 = target_quantile.shape[1]
    x = target_quantile.unsqueeze(1) - predicted_quantile.unsqueeze(2)
    assert x.shape == (batch_size, size1, size2)
    # huber = F.huber_loss(x, th.zeros_like(x), reduction='none')  # 这里使用huber loss的分布有偏，思考
    weight = tau.unsqueeze(2) - x.le(0).float()
    assert weight.shape == (batch_size, size1, size2)
    loss = (weight * x).mean(dim=(0,)).sum(dim=(0, 1))
    return loss