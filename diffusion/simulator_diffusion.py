import torch
import numpy as np
from sbi.utils import BoxUniform
import matplotlib.pyplot as plt
import scipy.signal as signal
#import numpy as np

T1 = torch.tensor( 1.0 )
T2 = torch.tensor( 2.0 )
d_wall = torch.tensor( 5.0 )
PROFILE_NOISE = 0.1
MEASUREMENT_NOISE = 0.01

##########################################################################
# single measurement

def diffusion_from_theta_single(theta):
    single_sample = False
    if theta.dim() == 1:
        single_sample = True
        theta = theta.unsqueeze(0)
    z = profile_from_theta_single(theta)
    x = measurement_from_profile_single( z  )
    return x

def diffusion_from_theta_with_latent_single(theta, T1 = T1):
    single_sample = False
    if theta.dim() == 1:
        single_sample = True
        theta = theta.unsqueeze(0)
    z = profile_from_theta_single(theta, T1 = T1)
    x = measurement_from_profile_single( z , T1 = T1 )
    return torch.concatenate( (x, z ), dim=1)


def profile_from_theta_single(theta, T1 = T1):
    single_sample = False
    if theta.dim() == 1:
        single_sample = True
        theta = theta.unsqueeze(0)
    diff_coef = theta[:,0]
    drift     = theta[:,1]
    #d_wall    = theta[:,2]
    #M_total   = theta[:,2]
    #T1 = 1.0
    #T2 = 2.0
    mean_1  = drift * T1
    sigma_1 = torch.sqrt(2.0*diff_coef * T1)
    #mean_2  = drift * T2
    #sigma_2 = torch.sqrt(2.0*diff_coef * T2 )
    z = torch.stack( [mean_1, sigma_1 ], dim=1  )
    eps = torch.randn_like(z) * PROFILE_NOISE
    z = z + eps

    return z


def measurement_from_profile_single( z , T1 = T1 ):
    if z.dim() == 1:
        single_sample = True
        z = z.unsqueeze(0)
    measurement_1 =  0.5 * (1.0 - torch.erf( (d_wall -  z[:,0]) / z[:,1]  )  )  
    #measurement_2 =  0.5 * (1.0 - torch.erf( (d_wall -  z[:,2]) / z[:,3]  )  )  
    #x = torch.stack( (measurement_1 , measurement_2) , dim=1  )
    x = torch.unsqueeze( measurement_1  , dim=1  )
    eps = torch.randn_like(x)*MEASUREMENT_NOISE
    x = x + eps
    return x



##########################################################################
# two measurements

def diffusion_from_theta(theta):
    single_sample = False
    if theta.dim() == 1:
        single_sample = True
        theta = theta.unsqueeze(0)
    z = profile_from_theta(theta)
    x = measurement_from_profile( z  )
    return x

def diffusion_from_theta_with_latent(theta):
    single_sample = False
    if theta.dim() == 1:
        single_sample = True
        theta = theta.unsqueeze(0)
    z = profile_from_theta(theta)
    x = measurement_from_profile( z  )
    return torch.concatenate( (x, z ), dim=1)


def profile_from_theta(theta):
    single_sample = False
    if theta.dim() == 1:
        single_sample = True
        theta = theta.unsqueeze(0)
    diff_coef = theta[:,0]
    drift     = theta[:,1]
    #d_wall    = theta[:,2]
    #M_total   = theta[:,2]
    T1 = 1.0
    T2 = 2.0
    mean_1  = drift * T1
    sigma_1 = torch.sqrt(2.0*diff_coef * T1)
    mean_2  = drift * T2
    sigma_2 = torch.sqrt(2.0*diff_coef * T2 )
    z = torch.stack( [mean_1, sigma_1, mean_2, sigma_2 ], dim=1  )

    return z


def measurement_from_profile( z  ):
    if z.dim() == 1:
        single_sample = True
        z = z.unsqueeze(0)
    measurement_1 =  0.5 * (1.0 - torch.erf( (d_wall -  z[:,0]) / z[:,1]  )  )  
    measurement_2 =  0.5 * (1.0 - torch.erf( (d_wall -  z[:,2]) / z[:,3]  )  )  
    x = torch.stack( (measurement_1 , measurement_2) , dim=1  )
    return x













