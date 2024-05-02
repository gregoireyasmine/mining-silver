import argparse
import os
import pickle
import time
from utils.snpe_utils import train_inferer

parser = argparse.ArgumentParser()

parser.add_argument('sim_number', metavar='n_sim', type=int, help="number of simulations to train the inferer on")
parser.add_argument('training_round', metavar='round', type=int, help="training session index")
args = parser.parse_args()
N_SIM = args.sim_number
ROUND_NB = args.training_round

ROOT = os.getcwd() + '/..'
SIM_PATH = os.path.join(ROOT, 'simulations/two_moons/100000sims_theta_z_x.pickle')
SAVE_MODEL_DIR = os.path.join(ROOT, 'validation/two_moons')
if not os.path.exists(SAVE_MODEL_DIR):
    os.makedirs(SAVE_MODEL_DIR)

with open(SIM_PATH, 'rb') as handle:
    all_simulations = pickle.load(handle)

theta_samples, z_samples, x_samples = all_simulations

theta = theta_samples[:N_SIM]
x = x_samples[:N_SIM]


print(f'round {ROUND_NB} of training standard NPE with simulation budget : {N_SIM} \n')
t1 = time.time()

print('theta:', theta, 'x:', x)
theta_results = train_inferer(theta, x, design='nsf')
t2 = time.time()
print(f'standard NPE achieved in {t2 - t1} seconds \n')


with open(SAVE_MODEL_DIR+f'/round_no_{ROUND_NB}_{N_SIM}_sim_standard_theta_results.pickle', 'wb') as handle:
    pickle.dump(theta_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

