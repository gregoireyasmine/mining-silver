import argparse
import os
from utils.snpe_utils import sample_for_observation
from utils.two_step_utils import two_step_sampling_from_obs
from utils.sliced_wasserstein import sliced_wasserstein_distance
from utils.pe_distances import pe_distance
import torch
from sbi.utils.metrics import c2st
import pickle
import time

parser = argparse.ArgumentParser()

parser.add_argument('name1', metavar='n1', type=str, help="pickle filename of the 1st posterior of theta (no suffix)")
parser.add_argument('name2', metavar='n2', type=str, help="pickle filename of the 2nd posterior of theta (np suffix)")
parser.add_argument('method', metavar='method', type=str, help="""Can be either 'c2st' or 'wasserstein' method""")
parser.add_argument('obs_path', metavar='obs_path', type=str, help="Path to the observations to evaluate posterior on")
parser.add_argument('num_samples', metavar='n_samples', type=int, help="Number of samples from each evaluated posterior")
parser.add_argument('save_name', metavar='savename', type=str, help="Name of the distance file (no suffix)")

args = parser.parse_args()

THETA1_NAME = args.name1
THETA2_NAME = args.name2
METHOD = args.method

NUM_SAMPLES = args.num_samples  # Number of samples used to estimate the distribution

SAVE_NAME = args.save_name

ROOT = os.getcwd()
MODEL_DIR = os.path.join(ROOT, 'validation', 'two_moons')
RESULTS_DIR = os.path.join(ROOT, 'results', 'multi_obs_distances')
SIM_PATH = args.obs_path

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

with open(SIM_PATH, 'rb') as handle:  # load observations
    observation_sims = pickle.load(handle)
true_thetas, true_z, x_obs = observation_sims

path_to_theta_1 = os.path.join(MODEL_DIR, THETA1_NAME+'.pickle')
with open(path_to_theta_1, 'rb') as handle:  # load posteriors
    results = pickle.load(handle)
    if len(results) == 2:
        estimator1 = (results[0][2], results[1][2])
    elif len(results) == 3:
        estimator1 = (results[2], )

path_to_theta_2 = os.path.join(MODEL_DIR, THETA2_NAME + '.pickle')
with open(path_to_theta_2, 'rb') as handle:  # load posteriors
    results = pickle.load(handle)
    if len(results) == 2:
        estimator2 = (results[0][2], results[1][2])
    elif len(results) == 3:
        estimator2 = (results[2], )
    else:
        raise Exception('Unknown estimator format')

t1 = time.time()
distances = pe_distance(estimator1, estimator2, METHOD, observations=true_thetas, num_samples=NUM_SAMPLES)

t2 = time.time()
print(f'computed distances between {THETA1_NAME} and {THETA2_NAME} in {t2-t1} s')

with open(os.path.join(RESULTS_DIR, f'{SAVE_NAME}.pickle'), 'wb') as handle:
    pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)

