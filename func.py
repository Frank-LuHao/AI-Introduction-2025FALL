import torch.nn.functional as F
import torch as th

def qtd_loss(predicted_quantile, target_quantile, tau):
    shape = predicted_quantile.shape
    x = target_quantile.unsqueeze(1) - predicted_quantile.unsqueeze(2)
    assert x.shape == (shape[0], shape[1], shape[1])
    # huber = F.huber_loss(x, th.zeros_like(x), reduction='none')  # 这里使用huber loss的分布有偏，思考
    weight = tau.unsqueeze(0).unsqueeze(2) - x.le(0).float()
    assert weight.shape == (shape[0], shape[1], shape[1])
    loss = (weight * x).mean(dim=(0)).sum(dim=(0, 1))
    return loss