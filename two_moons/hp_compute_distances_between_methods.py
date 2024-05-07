import argparse
import os
from tqdm import tqdm
from utils.snpe_utils import sample_for_observation
from utils.two_step_utils import simulate_two_step, two_step_sampling_from_obs
from utils.sliced_wasserstein import sliced_wasserstein_distance  # credit to Mackelab github
from two_moons import from_means, mean_function
from sbi.utils import BoxUniform
import torch
from sbi.utils.metrics import c2st
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('standard_theta_posterior_path', metavar='std_theta', type=int, help="path to the posterior of theta (standard est.)")
parser.add_argument('twostep_theta_posterior_path', metavar='2step_theta', type=int, help="path to the posterior of theta (2step est.)")
parser.add_argument('twostep_z_posterior_path', metavar='2step_theta', type=int, help="path to the posterior of z (2step est.)")

args = parser.parse_args()
STD_THETA_PATH = args.standard_theta_posterior_path
TSTP_THETA_PATH = args.twostep_theta_posterior_path
TSTP_Z_PATH = args.twostep_z_posterior_path


NUM_OBS = 100  # Number of "x observed" to average posterior distributions distances on
NUM_SAMPLES = 1000  # Number of samples used to estimate the distribution

ROOT = os.getcwd()
RESULTS_DIR = os.path.join(ROOT, 'results')
SIM_PATH = os.path.join(ROOT, f'simulations/two_moons/{NUM_OBS}sims_theta_z_x.pickle')

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

with open(SIM_PATH, 'rb') as handle:  # load observations
    observation_sims = pickle.load(handle)
true_thetas, true_z, x_obs = observation_sims

with open(STD_THETA_PATH, 'rb') as handle:  # load posteriors
    _, _, std_theta = pickle.load(handle)

with open(TSTP_THETA_PATH, 'rb') as handle:
    _, _, tstp_theta = pickle.load(handle)

with open(TSTP_Z_PATH, 'rb') as handle:
    _, _, tstp_z = pickle.load(handle)

c2st_distances = []
wasserstein_distances = []

for k in tqdm(range(NUM_OBS)):
    x_o = x_obs[k]
    std_post_samples = sample_for_observation(std_theta_posterior, x_o, n_post_samples=NUM_SAMPLES)
    _, twostep_post_samples = two_step_sampling_from_obs(twostep_z_posterior, twostep_theta_posterior, x_o, int(NUM_SAMPLES**0.5)+1, int(NUM_SAMPLES**0.5)+1)
    twostep_post_samples = twostep_post_samples[torch.randperm(len(twostep_post_samples))[:NUM_SAMPLES]]
    c2st_distances.append(c2st(twostep_post_samples, std_post_samples))
    wasserstein_distances.append(sliced_wasserstein_distance(twostep_post_samples, std_post_samples))
