import argparse
import os
from utils.snpe_utils import sample_for_observation
from utils.two_step_utils import two_step_sampling_from_obs
from utils.sliced_wasserstein import sliced_wasserstein_distance  # credit to Mackelab github
import torch
from sbi.utils.metrics import c2st
import pickle
import time

parser = argparse.ArgumentParser()

parser.add_argument('standard_theta_posterior_name', metavar='std_theta', type=int, help=" pickle filename of the the posterior of theta (standard est.)")
parser.add_argument('twostep_theta_posterior_name', metavar='2step_theta', type=int, help="pickle filename of the posterior of theta (2step est.)")
parser.add_argument('twostep_z_posterior_name', metavar='2step_theta', type=int, help="pickle filename of the posterior of z (2step est.)")
parser.add_argument('method', metavar='method', type=str, help="""Can be either 'c2st' or 'wasserstein' method""")
parser.add_argument('num_obs', metavar='n_obs', type=str, help="Number of observations to evaluate posterior on")
parser.add_argument('num_samples', metavar='n_samples', type=str, help="Number of samples from each evaluated posterior")
args = parser.parse_args()

STD_THETA_NAME = args.standard_theta_posterior_name
TSTP_THETA_NAME = args.twostep_theta_posterior_name
TSTP_Z_NAME = args.twostep_z_posterior_name
METHOD = args.method

NUM_OBS = 100  # Number of "x observed" to average posterior distributions distances on
NUM_SAMPLES = 1000  # Number of samples used to estimate the distribution

ROOT = os.getcwd()
MODEL_DIR = os.path.join(ROOT, 'validation', 'two_moons')
RESULTS_DIR = os.path.join(ROOT, 'results', 'multi_obs_distances')
SIM_PATH = os.path.join(ROOT, f'simulations/two_moons/{NUM_OBS}sims_theta_z_x.pickle')

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

with open(SIM_PATH, 'rb') as handle:  # load observations
    observation_sims = pickle.load(handle)
true_thetas, true_z, x_obs = observation_sims

path_to_std_theta = os.path.join(MODEL_DIR, STD_THETA_NAME+'.pickle')
with open(path_to_std_theta, 'rb') as handle:  # load posteriors
    _, _, std_theta = pickle.load(handle)

path_to_tstp_theta = os.path.join(MODEL_DIR, TSTP_THETA_NAME+'.pickle')
with open(path_to_tstp_theta, 'rb') as handle:
    _, _, tstp_theta = pickle.load(handle)

path_to_tstp_z = os.path.join(MODEL_DIR, TSTP_Z_NAME+'.pickle')
with open(path_to_tstp_z, 'rb') as handle:
    _, _, tstp_z = pickle.load(handle)

t1 = time.time()
distances = []

for k in range(NUM_OBS):
    x_o = x_obs[k]
    std_post_samples = sample_for_observation(std_theta, x_o, n_post_samples=NUM_SAMPLES)
    _, twostep_post_samples = two_step_sampling_from_obs(tstp_z, tstp_theta, x_o, int(NUM_SAMPLES**0.5)+1, int(NUM_SAMPLES**0.5)+1)
    twostep_post_samples = twostep_post_samples[torch.randperm(len(twostep_post_samples))[:NUM_SAMPLES]]
    if METHOD == 'c2st':
        distances.append(c2st(twostep_post_samples, std_post_samples))
    else:
        distances.append(sliced_wasserstein_distance(twostep_post_samples, std_post_samples))

t2 = time.time()
print(f'computed distances between {TSTP_THETA_NAME} and {STD_THETA_NAME} in {t2-t1} s')

with open(os.path.join(RESULTS_DIR, f'{METHOD}_distance_{STD_THETA_NAME}_vs_{TSTP_THETA_NAME}_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle'), 'wb') as handle:
    pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)
