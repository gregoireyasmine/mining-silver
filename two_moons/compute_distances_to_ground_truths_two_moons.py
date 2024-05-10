import pickle
import numpy as np
import os
import multiprocessing
import subprocess

os.chdir('../')
ROOT = os.getcwd()
DISTANCES_DIR = os.path.join(ROOT, 'results', 'multi_obs_distances')
RESULTS_DIR =  os.path.join(ROOT, 'results', 'mean_distances')
MODELS_DIR = os.path.join(ROOT, 'validation/two_moons')
NUM_OBS = 10  # Number of x_o to average posterior distributions distances on
NUM_SAMPLES = 10000  # Number of samples to compute the distance
SIM_BUDGETS = [100, 200, 300 , 500, 1000, 2000, 3000, 5000, 10000, 20000]  # Sim budgets on which inferers were trained
INFERER_NB  = [8,   8,   8 ,   8,   8,    8,    8,    8,    3,     3]  # Corresponding number of trained inferers

STD_TRUTH_NAME = f'round_no_{1}_{50_000}_sim_standard_theta_results'
TSTP_TRUTH_NAME_THETA = f'round_no_{1}_{50_000}_sim_twostep_theta_results'
TSTP_TRUTH_NAME_Z = f'round_no_{1}_{50_000}_sim_twostep_z_results'


def run_script(args):
    python_script, params = args
    command = f"python {python_script} {params}"
    subprocess.run(command, shell=True)


scripts_and_params = []
# dist std to std gd truth
for i, n_sim in enumerate(SIM_BUDGETS):
    for nb in range(1, INFERER_NB[i]+1):
        fnm = f'round_no_{nb}_{n_sim}_sim_standard_theta_results'
        for method in ['c2st', 'wasserstein']:
            script_and_params = ('two_moons/hp_compute_distances_standard.py', f"{fnm} {STD_TRUTH_NAME} {method} {NUM_OBS} {NUM_SAMPLES}")
            scripts_and_params.append(script_and_params)


# dist std to 2step gd truth
for i, n_sim in enumerate(SIM_BUDGETS):
    for nb in range(1, INFERER_NB[i]+1):
        fnm = f'round_no_{nb}_{n_sim}_sim_standard_theta_results'
        for method in ['c2st', 'wasserstein']:
            script_and_params = ('two_moons/hp_compute_distances_between_methods.py', f"{fnm} {TSTP_TRUTH_NAME_THETA} {TSTP_TRUTH_NAME_Z} {method} {NUM_OBS} {NUM_SAMPLES}")
            scripts_and_params.append(script_and_params)


# dist 2step to 2step gd truth
for i, n_sim in enumerate(SIM_BUDGETS):
    for nb in range(1, INFERER_NB[i]+1):
        fnm_theta = f'round_no_{nb}_{n_sim}_sim_twostep_theta_results'
        fnm_z = f'round_no_{nb}_{n_sim}_sim_twostep_z_results'
        for method in ['c2st', 'wasserstein']:
            script_and_params = ('two_moons/hp_compute_distances_twostep.py', f"{fnm_theta} {fnm_z} {TSTP_TRUTH_NAME_THETA} {TSTP_TRUTH_NAME_Z} {method} {NUM_OBS} {NUM_SAMPLES}")
            scripts_and_params.append(script_and_params)

# dist 2step to std gd truth
for i, n_sim in enumerate(SIM_BUDGETS):
    for nb in range(1, INFERER_NB[i]+1):
        fnm_theta = f'round_no_{nb}_{n_sim}_sim_twostep_theta_results'
        fnm_z = f'round_no_{nb}_{n_sim}_sim_twostep_z_results'
        for method in ['c2st', 'wasserstein']:
            script_and_params = ('two_moons/hp_compute_distances_between_methods.py', f"{STD_TRUTH_NAME} {fnm_theta} {fnm_z} {method} {NUM_OBS} {NUM_SAMPLES}")
            scripts_and_params.append(script_and_params)

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=2)
    pool.map(run_script, scripts_and_params)

    for k, ground_truth_name in enumerate([STD_TRUTH_NAME, TSTP_TRUTH_NAME_THETA]):
        for m, inferer_type in enumerate(['standard', 'twostep']):
            for method in ['c2st', 'wasserstein']:
                all_distances = []
                for i, n_sim in enumerate(SIM_BUDGETS):
                    avg_distances = []
                    for nb1 in range(1, INFERER_NB[i]+1):
                        fnm = f'round_no_{nb1}_{n_sim}_sim_{inferer_type}_theta_results'
                        if m > k:
                            distance_file = f'{method}_distance_{ground_truth_name}_vs_{fnm}_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle'
                        else:
                            distance_file = f'{method}_distance_{fnm}_vs_{ground_truth_name}_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle'
                        with open(os.path.join(DISTANCES_DIR, distance_file), 'rb') as handle:
                            distances = pickle.load(handle)
                        avg_distances.append(np.mean(distances))
                    all_distances.append(avg_distances)
                with open(os.path.join(RESULTS_DIR, f'mean_{method}_distances_{inferer_type}_inferers_to_{ground_truth_name}_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle'), 'wb') as handle:
                    pickle.dump(all_distances, handle, protocol=pickle.HIGHEST_PROTOCOL)
