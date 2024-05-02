import argparse
import os
import pickle
import time
from snpe_utils import train_inferer
import torch

parser = argparse.ArgumentParser()

parser.add_argument('sim_number', metavar='n_sim', type=int, help="number of simulations to train the inferer on")
parser.add_argument('training_round', metavar='round', type=int, help="training session index")
args = parser.parse_args()
n_sim = args.sim_number
round_nb = args.training_round

root = os.getcwd()
sim_path = os.path.join(root, 'simulations/two_moons/100000sims_theta_z_x.pickle')


with open(sim_path, 'rb') as handle:
    all_simulations = pickle.load(handle)

theta_samples, z_samples, x_samples = all_simulations
theta = theta_samples[:n_sim]
z_samples = z_samples[:n_sim]
x_samples = x_samples[:n_sim]


print('training 2step NPE with simulation budget : {n_sim} \n')
t1 = time.time()
z_results = train_inferer(z_samples, x_samples, design='nsf')
theta_results = train_inferer(theta_samples, torch.concatenate([x_samples, z_samples], dim=1), design='nsf')
t2 = time.time()
print(f'2step NPE achieved in {t2 - t1} seconds \n')


with open(f'validation/round_no_{round_nb}_{n_sim}_sim_twostep_z_results.pickle', 'wb') as handle:
    pickle.dump(z_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'validation/round_no_{round_nb}_{n_sim}_sim_twostep_theta_results.pickle', 'wb') as handle:
    pickle.dump(theta_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

