import pickle
from tqdm import tqdm
from snpe_utils import sample_for_observation
from two_step_utils import simulate_two_step, two_step_sampling_from_obs
from two_moons import from_means, mean_function
from sbi.utils import BoxUniform
import torch
from sbi.utils.metrics import c2st
from sliced_wasserstein import sliced_wasserstein_distance  # credit to Mackelab github
import numpy as np
import os

print(os.getcwd())
NUM_OBS = 100  # Number of x_o to average posterior distributions distances on
NUM_SAMPLES = 1000  # Number of samples to compute the
SIM_BUDGETS = [10, 20, 50, 75, 100, 200, 300, 500, 1000, 2000, 10000, 100000]

theta_prior = BoxUniform(low=torch.tensor([-1.0, -1.0]), high=torch.tensor([1.0, 1.0]))

mean_c2st_2_methods = []
mean_wasserstein_2_methods = []

true_thetas, z_obs, x_obs = simulate_two_step(mean_function, from_means, theta_prior, NUM_OBS)

## Distribution convergence plots

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
