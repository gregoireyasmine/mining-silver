import argparse
import os
from utils.two_step_utils import two_step_sampling_from_obs
from utils.sliced_wasserstein import sliced_wasserstein_distance  # credit to Mackelab github
import torch
from sbi.utils.metrics import c2st
import pickle
import time

parser = argparse.ArgumentParser()

parser.add_argument('theta_name_1', metavar='theta1', type=int, help="filename of the 1st posterior of theta")
parser.add_argument('z_name_1', metavar='z1', type=int, help="filename of the 1st posterior of z ")
parser.add_argument('theta_name_2', metavar='theta2', type=int, help="filename of the 2nd posterior of theta")
parser.add_argument('z_name_2', metavar='z2', type=int, help="filename of the 2nd posterior of z")
parser.add_argument('method', metavar='method', type=str, help="""Can be either 'c2st' or 'wasserstein' method""")
parser.add_argument('num_obs', metavar='n_obs', type=str, help="Number of observations to evaluate posterior on")
parser.add_argument('num_samples', metavar='n_samples', type=str, help="Number of samples from each evaluated posterior")

args = parser.parse_args()

THETA_NAME_1 = args.theta_name_1
THETA_NAME_2 = args.theta_name_2
Z_NAME_1 = args.z_name_1
Z_NAME_2 = args.z_name_2
METHOD = args.method
NUM_OBS = args.num_obs
NUM_SAMPLES = args.num_samples

ROOT = os.getcwd()
RESULTS_DIR = os.path.join(ROOT, 'results', 'multi_obs_distances')
SIM_PATH = os.path.join(ROOT, f'simulations/two_moons/{NUM_OBS}sims_theta_z_x.pickle')
MODEL_DIR = os.path.join(ROOT, 'validation', 'two_moons')

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

with open(SIM_PATH, 'rb') as handle:  # load observations
    observation_sims = pickle.load(handle)
true_thetas, true_z, x_obs = observation_sims

path_to_theta_1 = os.path.join(MODEL_DIR, THETA_NAME_1+'.pickle')
with open(path_to_theta_1, 'rb') as handle:
    _, _, theta_1 = pickle.load(handle)

path_to_z_1 = os.path.join(MODEL_DIR, Z_NAME_1+'.pickle')
with open(path_to_z_1, 'rb') as handle:
    _, _, z_1 = pickle.load(handle)

path_to_theta_2 = os.path.join(MODEL_DIR, THETA_NAME_2+'.pickle')
with open(THETA_NAME_2, 'rb') as handle:
    _, _, theta_2 = pickle.load(handle)

path_to_z_2 = os.path.join(MODEL_DIR, Z_NAME_2+'.pickle')
with open(path_to_z_2, 'rb') as handle:
    _, _, z_2 = pickle.load(handle)

distances = []

t1 = time.time()
for k in range(NUM_OBS):
    x_o = x_obs[k]
    _, samples1 = two_step_sampling_from_obs(z_1, theta_1, x_o, int(NUM_SAMPLES**0.5)+1, int(NUM_SAMPLES**0.5)+1)
    samples1 = samples1[torch.randperm(len(samples1))[:NUM_SAMPLES]]
    _, samples2 = two_step_sampling_from_obs(z_2, theta_2, x_o, int(NUM_SAMPLES ** 0.5) + 1,
                                             int(NUM_SAMPLES ** 0.5) + 1)
    samples2 = samples2[torch.randperm(len(samples2))[:NUM_SAMPLES]]
    if METHOD == 'c2st':
        distances.append(c2st(samples1, samples2))
    else:
        distances.append(sliced_wasserstein_distance(samples1, samples2))
t2 = time.time()

print(f'computed distances between {THETA_NAME_1} and {THETA_NAME_2} in {t2-t1} s')

with open(os.path.join(RESULTS_DIR, f'{METHOD}_distance_{THETA_NAME_1}_vs_{THETA_NAME_2}_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle'), 'wb') as handle:
    pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)
