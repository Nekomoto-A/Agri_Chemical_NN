import torch
def gaussian_kernel(x, y, sigma=1.0):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist_sq = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return torch.exp(-dist_sq / (2 * sigma ** 2))

def mmd_rbf(source, target, sigma=1.0):
    if len(source) == 0 or len(target) == 0:
        return torch.tensor(0.0, device=source.device) # どちらかのグループが空なら損失は0
    
    k_ss = gaussian_kernel(source, source, sigma).mean()
    k_tt = gaussian_kernel(target, target, sigma).mean()
    k_st = gaussian_kernel(source, target, sigma).mean()
    
    mmd_loss = k_ss + k_tt - 2 * k_st
    return mmd_loss