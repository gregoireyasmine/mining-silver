import pickle
from tqdm import tqdm
import numpy as np
import os
import multiprocessing
import subprocess

os.chdir('../')
ROOT = os.getcwd()
DISTANCES_DIR = os.path.join(ROOT, 'results', 'multi_obs_distances')
RESULTS_DIR =  os.path.join(ROOT, 'results', 'mean_distances')
MODELS_DIR = os.path.join(ROOT, 'validation/two_moons')
NUM_OBS = 100  # Number of x_o to average posterior distributions distances on
NUM_SAMPLES = 5000  # Number of samples to compute the distance
SIM_BUDGETS = [100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000, 20000]  # Sim budgets on which inferers were trained
INFERER_NB  = [8,   8,   8,   8,   8,    8,    8,    8,    3,     3]  # Corresponding number of trained inferers


def run_script(args):
    python_script, params = args
    command = f"python {python_script} {params}"
    subprocess.run(command, shell=True)


scripts_and_params = []
for i, n_sim in tqdm(enumerate(SIM_BUDGETS)):
    for nb1 in range(1, INFERER_NB[i]+1):
        for nb2 in range(nb1+1, INFERER_NB[i]):
            filename1 = f'round_no_{nb1}_{n_sim}_sim_std_theta_results'
            filename2 = f'round_no_{nb2}_{n_sim}_sim_std_theta_results'
            for method in ['c2st', 'wasserstein']:
                script_and_params = ('two_moons/hp_compute_distances_standard.py', f"{filename1} {filename2} {method} {NUM_OBS} {NUM_SAMPLES}")
            scripts_and_params.append(script_and_params)

for i, n_sim in tqdm(enumerate(SIM_BUDGETS)):
    for nb1 in range(INFERER_NB[i]):
        for nb2 in range(nb1+1, INFERER_NB[i]+1):
            theta_fnm_1 = f'round_no_{nb1}_{n_sim}_sim_twostep_theta_results'
            theta_fnm_2 = f'round_no_{nb2}_{n_sim}_sim_twostep_theta_results'
            z_fnm_1 = f'round_no_{nb1}_{n_sim}_sim_twostep_z_results'
            z_fnm_2 = f'round_no_{nb2}_{n_sim}_sim_twostep_z_results'
            for method in ['c2st', 'wasserstein']:
                script_and_params = ('two_moons/hp_compute_distances_twostep.py', f"{theta_fnm_1} {z_fnm_1} {theta_fnm_2} {z_fnm_2} {method} {NUM_OBS} {NUM_SAMPLES}")
            scripts_and_params.append(script_and_params)

pool = multiprocessing.Pool(processes=len(scripts_and_params))
pool.map(run_script, scripts_and_params)

for inferer_type in ['std', 'twostep']:
    for method in ['c2st', 'wasserstein']:
        all_distances = []
        for i, n_sim in enumerate(SIM_BUDGETS):
            avg_distances = []
            for nb1 in range(INFERER_NB[i]):
                for nb2 in range(nb1+1, INFERER_NB[i]):
                    fnm_1 = f'round_no_{nb1}_{n_sim}_sim_{inferer_type}_theta_results'
                    fnm_2 = f'round_no_{nb2}_{n_sim}_sim_{inferer_type}_theta_results'
                    std_distance_file = f'{method}_distance_{fnm_1}_vs_{fnm_2}_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle'
                    with open(os.path.join(DISTANCES_DIR, std_distance_file), 'rb') as handle:
                        distances = pickle.load(handle)
                    avg_distances.append(np.mean(distances))
            all_distances.append(avg_distances)
        with open(os.path.join(RESULTS_DIR, f'mean_{method}_distances_between_{inferer_type}_inferers_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle'), 'wb') as handle:
            pickle.dump(all_distances, handle, protocol=pickle.HIGHEST_PROTOCOL)
