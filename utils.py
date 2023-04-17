import numpy as np
import torch

def distance(x_adv, x, p=2):
    diff = x_adv - x
    # diff_flat = diff.reshape(diff.shape[0], -1)
    diff_flat = diff.flatten()
    return np.linalg.norm(diff_flat, p)

def torch_distance(x_adv, x, p='2'):
    if x is None:
        diff = x_adv.reshape(x_adv.size(0), -1)
    else:
        diff = (x_adv - x).reshape(x.size(0), -1)
    if p == '2':
        out = torch.sqrt(torch.sum(diff * diff)).item()
    elif p == 'inf':
        out = torch.sum(torch.max(torch.abs(diff), 1)[0]).item()
    return out
