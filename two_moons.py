import torch
import numpy as np
from sbi.utils import BoxUniform


# This is a slight modification of the "Two Moons" simulator - see https://arxiv.org/abs/1905.07488 for details.
def modified_TM(theta):
    single_sample = False
    if theta.dim() == 1:
        single_sample = True
        theta = theta.unsqueeze(0)
    means = mean_function(theta)
    alpha = BoxUniform(low=torch.Tensor([-np.pi/2]), high=torch.Tensor([np.pi/2])).sample(((theta.shape[0]),)).squeeze()
    r = 0.01*torch.randn_like(alpha) + 0.1
    x = torch.stack([r*torch.cos(alpha) + means[:, 0]+0.25, r*torch.sin(alpha) + means[:, 1]], dim=1)
    return x


# Mean function with some added noise
def mean_function(theta):
    means = torch.stack([torch.abs(theta[:, 0]+ theta[:, 1])/np.sqrt(2),(-theta[:,0] + theta[:, 1])/np.sqrt(2)], dim=1)
    eps = torch.randn_like(means)*0.01
    means += eps
    return means


# Simulating directly from the intermediate variables (means)
def from_means(means):
    alpha = BoxUniform(low=torch.Tensor([-np.pi/2]), high=torch.Tensor([np.pi/2])).sample(((means.shape[0]),)).squeeze()
    r = 0.01*torch.randn_like(alpha) + 0.1
    x = torch.stack([r*torch.cos(alpha) + means[:, 0]+0.25, r*torch.sin(alpha) + means[:, 1]], dim=1)
    return x
