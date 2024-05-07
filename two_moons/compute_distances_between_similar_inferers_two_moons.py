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
            filename1 = os.path.join(MODELS_DIR, f'round_no_{nb1}_{n_sim}_sim_std_theta_results')
            filename2 = os.path.join(MODELS_DIR, f'round_no_{nb2}_{n_sim}_sim_std_theta_results')
            for method in ['c2st', 'wasserstein']:
                script_and_params = ('two_moons/hp_compute_distances_standard.py', f"{filename1} {filename2} {method} {NUM_OBS} {NUM_SAMPLES}")
            scripts_and_params.append(script_and_params)

for i, n_sim in tqdm(enumerate(SIM_BUDGETS)):
    for nb1 in range(INFERER_NB[i]):
        for nb2 in range(nb1+1, INFERER_NB[i]+1):
            theta_fnm_1 = os.path.join(MODELS_DIR, f'round_no_{nb1}_{n_sim}_sim_twostep_theta_results')
            theta_fnm_2 = os.path.join(MODELS_DIR, f'round_no_{nb2}_{n_sim}_sim_twostep_theta_results')
            z_fnm_1 = os.path.join(MODELS_DIR, f'round_no_{nb1}_{n_sim}_sim_twostep_z_results')
            z_fnm_2 = os.path.join(MODELS_DIR, f'round_no_{nb2}_{n_sim}_sim_twostep_z_results')
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
                    for method in ['c2st', 'wasserstein']:
                        std_distance_file = f'{method}_distance_{fnm_1}_vs_{fnm_2}_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle'
                        with open(os.path.join(DISTANCES_DIR, std_distance_file), 'rb') as handle:
                            distances = pickle.load(handle)
                            avg_distances.append(np.mean(distances))
            all_distances.append(avg_distances)
        with open(os.path.join(RESULTS_DIR, f'mean_{method}_distances_between_{inferer_type}_inferers_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle'), 'wb') as handle:
            pickle.dump(all_distances, handle, protocol=pickle.HIGHEST_PROTOCOL)


## Distribution convergence plots
'''
for n_sim in tqdm(SIM_BUDGETS):

    with open('validation/' + str(n_sim) + '_sim_std_theta_posterior.pickle', 'rb') as handle:
        std_theta_posterior = pickle.load(handle)
    with open('validation/' + str(n_sim) + '_sim_2_step_Z_posterior.pickle', 'rb') as handle:
        twostep_z_posterior = pickle.load(handle)
    with open('validation/' + str(n_sim) + '_sim_2_step_theta_posterior.pickle', 'rb') as handle:
        twostep_theta_posterior = pickle.load(handle)

    # sample 10k from posterior
    c2st_distances = []
    wasserstein_distances = []

    for k in range(NUM_OBS):
        x_o = x_obs[k]
        std_post_samples = sample_for_observation(std_theta_posterior, x_o, n_post_samples=NUM_SAMPLES)
        _, twostep_post_samples = two_step_sampling_from_obs(twostep_z_posterior, twostep_theta_posterior, x_o, int(NUM_SAMPLES**0.5)+1, int(NUM_SAMPLES**0.5)+1)
        twostep_post_samples = twostep_post_samples[torch.randperm(len(twostep_post_samples))[:NUM_SAMPLES]]
        c2st_distances.append(c2st(twostep_post_samples, std_post_samples))
        wasserstein_distances.append(sliced_wasserstein_distance(twostep_post_samples, std_post_samples))

    mean_c2st_2_methods.append(np.mean(c2st_distances))
    mean_wasserstein_2_methods.append(np.mean(wasserstein_distances))

with open(f'results/mean_c2st_distance_between_methods_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'wb') as handle:
    pickle.dump(mean_c2st_2_methods, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'results/mean_wasserstein_distance_between_methods_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'wb') as handle:
    pickle.dump(mean_wasserstein_2_methods, handle, protocol=pickle.HIGHEST_PROTOCOL)


## Convergence toward ground truth plots

with open('validation/'+str(100_000)+'_sim_std_theta_posterior.pickle', 'rb') as handle:
    ground_truth_standard = pickle.load(handle)
with open('validation/'+str(100_000)+'_sim_2_step_theta_posterior.pickle', 'rb') as handle:
    ground_truth_theta_twostep = pickle.load(handle)
with open('validation/' + str(100_000) + '_sim_2_step_Z_posterior.pickle', 'rb') as handle:
    ground_truth_z_twostep = pickle.load(handle)

mean_twostep_to_twostep_ground_c2st_distances = []
mean_twostep_to_std_ground_c2st_distances = []
mean_std_to_twostep_ground_c2st_distances = []
mean_std_to_std_ground_c2st_distances = []
mean_twostep_to_twostep_ground_wasserstein_distances = []
mean_twostep_to_std_ground_wasserstein_distances = []
mean_std_to_twostep_ground_wasserstein_distances = []
mean_std_to_std_ground_wasserstein_distances = []

for n_sim in tqdm(SIM_BUDGETS):

    with open('validation/' + str(n_sim) + '_sim_std_theta_posterior.pickle', 'rb') as handle:
        std_theta_posterior = pickle.load(handle)
    with open('validation/' + str(n_sim) + '_sim_2_step_Z_posterior.pickle', 'rb') as handle:
        twostep_z_posterior = pickle.load(handle)
    with open('validation/' + str(n_sim) + '_sim_2_step_theta_posterior.pickle', 'rb') as handle:
        twostep_theta_posterior = pickle.load(handle)

    # sample 10k from posterior
    twostep_to_twostep_ground_c2st_distances = []
    twostep_to_std_ground_c2st_distances = []
    std_to_twostep_ground_c2st_distances = []
    std_to_std_ground_c2st_distances = []
    twostep_to_twostep_ground_wasserstein_distances = []
    twostep_to_std_ground_wasserstein_distances = []
    std_to_twostep_ground_wasserstein_distances = []
    std_to_std_ground_wasserstein_distances = []

    for k in range(NUM_OBS):
        x_o = x_obs[k]

        std_ground_truth_samples = sample_for_observation(ground_truth_standard, x_o, n_post_samples=NUM_SAMPLES)
        _, twostep_ground_truth_samples = two_step_sampling_from_obs(ground_truth_z_twostep, ground_truth_theta_twostep, x_o, int(NUM_SAMPLES**0.5)+1, int(NUM_SAMPLES**0.5)+1)
        twostep_ground_truth_samples = twostep_ground_truth_samples[torch.randperm(len(twostep_ground_truth_samples))[:NUM_SAMPLES]]

        std_post_samples = sample_for_observation(std_theta_posterior, x_o, n_post_samples=NUM_SAMPLES)
        _, twostep_post_samples = two_step_sampling_from_obs(twostep_z_posterior, twostep_theta_posterior, x_o, int(NUM_SAMPLES**0.5)+1, int(NUM_SAMPLES**0.5)+1)
        twostep_post_samples = twostep_post_samples[torch.randperm(len(twostep_post_samples))[:NUM_SAMPLES]]

        twostep_to_twostep_ground_c2st_distances.append(c2st(twostep_post_samples, twostep_ground_truth_samples))
        twostep_to_std_ground_c2st_distances.append(c2st(twostep_post_samples, std_ground_truth_samples))
        std_to_twostep_ground_c2st_distances.append(c2st(std_post_samples, twostep_ground_truth_samples))
        std_to_std_ground_c2st_distances.append(c2st(std_post_samples, std_ground_truth_samples))

        twostep_to_twostep_ground_wasserstein_distances.append(sliced_wasserstein_distance(twostep_post_samples, twostep_ground_truth_samples))
        twostep_to_std_ground_wasserstein_distances.append(sliced_wasserstein_distance(twostep_post_samples, std_ground_truth_samples))
        std_to_twostep_ground_wasserstein_distances.append(sliced_wasserstein_distance(std_post_samples, twostep_ground_truth_samples))
        std_to_std_ground_wasserstein_distances.append(sliced_wasserstein_distance(std_post_samples, std_ground_truth_samples))

    mean_twostep_to_twostep_ground_c2st_distances.append(np.mean(twostep_to_twostep_ground_c2st_distances))
    mean_twostep_to_std_ground_c2st_distances.append(np.mean(twostep_to_std_ground_c2st_distances))
    mean_std_to_twostep_ground_c2st_distances.append(np.mean(std_to_twostep_ground_c2st_distances))
    mean_std_to_std_ground_c2st_distances.append(np.mean(std_to_std_ground_c2st_distances))
    mean_twostep_to_twostep_ground_wasserstein_distances.append(np.mean(twostep_to_twostep_ground_wasserstein_distances))
    mean_twostep_to_std_ground_wasserstein_distances.append(np.mean(twostep_to_std_ground_wasserstein_distances))
    mean_std_to_twostep_ground_wasserstein_distances.append(np.mean(std_to_twostep_ground_wasserstein_distances))
    mean_std_to_std_ground_wasserstein_distances.append(np.mean(std_to_std_ground_wasserstein_distances))

with open(f'results/mean_twostep_to_twostep_ground_c2st_distances_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'wb') as handle:
    pickle.dump(mean_twostep_to_twostep_ground_c2st_distances, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'results/mean_twostep_to_std_ground_c2st_distances_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'wb') as handle:
    pickle.dump(mean_twostep_to_std_ground_c2st_distances, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'results/mean_std_to_twostep_ground_c2st_distances_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'wb') as handle:
    pickle.dump(mean_std_to_twostep_ground_c2st_distances, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'results/mean_std_to_std_ground_c2st_distances_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'wb') as handle:
    pickle.dump(mean_std_to_std_ground_c2st_distances, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'results/mean_twostep_to_twostep_ground_wasserstein_distances_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'wb') as handle:
    pickle.dump(mean_twostep_to_twostep_ground_wasserstein_distances, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'results/mean_twostep_to_std_ground_wasserstein_distances_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'wb') as handle:
    pickle.dump(mean_twostep_to_std_ground_wasserstein_distances, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'results/mean_std_to_twostep_ground_wasserstein_distances_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'wb') as handle:
    pickle.dump(mean_std_to_twostep_ground_wasserstein_distances, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'results/mean_std_to_std_ground_wasserstein_distances_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'wb') as handle:
    pickle.dump(mean_std_to_std_ground_wasserstein_distances, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''