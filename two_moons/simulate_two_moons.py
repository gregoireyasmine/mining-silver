from utils.two_step_utils import simulate_two_step
from two_moons import from_means, mean_function
import torch
import pickle
from sbi.utils import BoxUniform
import os

SIM_NUMBER = 100_000  # how many sims to run
ROOT = os.getcwd() + '/..'
SIMULATIONS_DIR = os.path.join(ROOT, 'simulations/two_moons')

if not os.path.exists(SIMULATIONS_DIR):
    os.makedirs(SIMULATIONS_DIR)

theta_prior = BoxUniform(low=torch.tensor([-1.0, -1.0]), high=torch.tensor([1.0, 1.0]))
theta, z_sim, x_sim = simulate_two_step(mean_function, from_means, theta_prior, SIM_NUMBER)

with open(SIMULATIONS_DIR+f'/{SIM_NUMBER}sims_theta_z_x.pickle', 'wb') as handle:
    pickle.dump((theta, z_sim, x_sim), handle, protocol=pickle.HIGHEST_PROTOCOL)
