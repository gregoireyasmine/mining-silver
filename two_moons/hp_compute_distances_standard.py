import argparse
import os
from tqdm import tqdm
from utils.snpe_utils import sample_for_observation
from utils.sliced_wasserstein import sliced_wasserstein_distance  # credit to Mackelab github
from sbi.utils.metrics import c2st
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('name1', metavar='n1', type=int, help="pickle filename of the 1st posterior of theta")
parser.add_argument('name2', metavar='n2', type=int, help="pickle filename of the 2nd posterior of theta")
parser.add_argument('method', metavar='method', type=str, help="""Can be either 'c2st' or 'wasserstein' method""")
parser.add_argument('num_obs', metavar='n_obs', type=str, help="Number of observations to evaluate posterior on")
parser.add_argument('num_samples', metavar='n_samples', type=str, help="Number of samples from each evaluated posterior")

args = parser.parse_args()

THETA_1_NAME = args.name1
THETA_2_NAME = args.name2
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

path_to_theta_1 = os.path.join(MODEL_DIR, THETA_1_NAME+'.pickle')
with open(path_to_theta_1, 'rb') as handle:  # load posteriors
    _, _, theta1 = pickle.load(handle)

path_to_theta_2 = os.path.join(MODEL_DIR, THETA_2_NAME+'.pickle')
with open(path_to_theta_2, 'rb') as handle:
    _, _, theta2 = pickle.load(handle)

distances = []

for k in tqdm(range(NUM_OBS)):
    x_o = x_obs[k]
    theta1_post_samples = sample_for_observation(theta1, x_o, n_post_samples=NUM_SAMPLES)
    theta2_post_samples = sample_for_observation(theta2, x_o, n_post_samples=NUM_SAMPLES)

    if METHOD == 'c2st':
        distances.append(c2st(theta1_post_samples, theta2_post_samples))
    else:
        distances.append(sliced_wasserstein_distance(theta1_post_samples, theta2_post_samples))

with open(os.path.join(RESULTS_DIR, f'{METHOD}_distance_{THETA_1_NAME}_vs_{THETA_2_NAME}_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle'), 'wb') as handle:
    pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)
