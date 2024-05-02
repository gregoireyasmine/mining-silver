import matplotlib.pyplot as plt
import pickle
from snpe_utils import sample_for_observation
from two_step_utils import two_step_sampling_from_obs, simulate_two_step
from two_moons import from_means, mean_function
from sbi.utils import BoxUniform
import torch

SIM_BUDGETS = [10, 20, 50, 75, 100, 200, 300, 500, 1000, 2000, 10000, 100000]
NUM_OBS = 100
NUM_SAMPLES = 1000


## Figure 1: evaluate distances between two methods over simulation budgets

with open(f'results/mean_c2st_distance_between_methods_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'rb') as handle:
    mean_c2st_2_methods = pickle.load(handle)

with open(f'results/mean_wasserstein_distance_between_methods_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'rb') as handle:
    mean_wasserstein_2_methods = pickle.load(handle)

fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
ax[0].semilogx(SIM_BUDGETS, mean_c2st_2_methods)
ax[0].set_title('C2ST distance')

ax[1].semilogx(SIM_BUDGETS, mean_wasserstein_2_methods)
ax[1].set_title('Sliced Wasserstein distance')

ax[0].set_xlabel('Number of simulations')
ax[1].set_xlabel('Number of simulations')
ax[0].set_ylabel('accuracy')
ax[1].set_ylabel('Wasserstein distance')

ax[0].set_ylim(0.5, 0.99)
fig.suptitle('Distance between posteriors, averaged over observations')

plt.tight_layout()
plt.savefig(f'figs/distances/distance_between_methods_{NUM_OBS}_obs_{NUM_SAMPLES}_samples')


## Figure 2: evaluate distances between method and standard ground truth

with open(f'results/mean_twostep_to_twostep_ground_c2st_distances_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'rb') as handle:
    twostep_twostep_c2st = pickle.load(handle)

with open(f'results/mean_twostep_to_std_ground_c2st_distances_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'rb') as handle:
    twostep_std_c2st = pickle.load(handle)

with open(f'results/mean_std_to_twostep_ground_c2st_distances_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'rb') as handle:
    std_twostep_c2st = pickle.load(handle)

with open(f'results/mean_std_to_std_ground_c2st_distances_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'rb') as handle:
    std_std_c2st = pickle.load(handle)

with open(f'results/mean_twostep_to_twostep_ground_wasserstein_distances_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'rb') as handle:
    twostep_twostep_wass = pickle.load(handle)

with open(f'results/mean_twostep_to_std_ground_wasserstein_distances_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'rb') as handle:
    twostep_std_wass = pickle.load(handle)

with open(f'results/mean_std_to_twostep_ground_wasserstein_distances_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'rb') as handle:
    std_twostep_wass = pickle.load(handle)

with open(f'results/mean_std_to_std_ground_wasserstein_distances_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'rb') as handle:
    std_std_wass = pickle.load(handle)

fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
ax[0].semilogx(SIM_BUDGETS, twostep_twostep_c2st, label='two-step NPE')
ax[0].semilogx(SIM_BUDGETS, std_twostep_c2st, label='standard NPE')
ax[0].set_title('C2ST distance')
ax[0].legend()
ax[1].semilogx(SIM_BUDGETS, twostep_twostep_wass)
ax[1].semilogx(SIM_BUDGETS, std_twostep_wass)

ax[1].set_title('Sliced Wasserstein distance')

ax[0].set_xlabel('Number of simulations')
ax[1].set_xlabel('Number of simulations')
ax[0].set_ylabel('accuracy')
ax[1].set_ylabel('Wasserstein distance')

ax[0].set_ylim(0.5, 0.99)
fig.suptitle('Distance to ground truth (twostep), averaged over observations')

plt.tight_layout()
plt.savefig(f'figs/distances/distance_to_twostep_limit_{NUM_OBS}_obs_{NUM_SAMPLES}_samples')


fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
ax[0].semilogx(SIM_BUDGETS, twostep_std_c2st, label='two-step NPE')
ax[0].semilogx(SIM_BUDGETS, std_std_c2st, label='standard NPE')
ax[0].set_title('C2ST distance')
ax[0].legend()
ax[1].semilogx(SIM_BUDGETS, twostep_std_wass)
ax[1].semilogx(SIM_BUDGETS, std_std_wass)

ax[1].set_title('Sliced Wasserstein distance')

ax[0].set_xlabel('Number of simulations')
ax[1].set_xlabel('Number of simulations')
ax[0].set_ylabel('accuracy')
ax[1].set_ylabel('Wasserstein distance')

ax[0].set_ylim(0.5, 0.99)
fig.suptitle('Distance to ground truth (standard), averaged over observations')

plt.tight_layout()
plt.savefig(f'figs/distances/distance_to_standard_limit_{NUM_OBS}_obs_{NUM_SAMPLES}_samples')

## Figure 3 : plot distributions
theta_prior = BoxUniform(low=torch.tensor([-1.0, -1.0]), high=torch.tensor([1.0, 1.0]))
true_theta, true_z, x_o = simulate_two_step(mean_function, from_means, theta_prior, 1)

for n_sim in SIM_BUDGETS:
    with open('validation/' + str(n_sim) + '_sim_std_theta_posterior.pickle', 'rb') as handle:
        std_theta_posterior = pickle.load(handle)
    with open('validation/' + str(n_sim) + '_sim_2_step_Z_posterior.pickle', 'rb') as handle:
        twostep_z_posterior = pickle.load(handle)
    with open('validation/' + str(n_sim) + '_sim_2_step_theta_posterior.pickle', 'rb') as handle:
        twostep_theta_posterior = pickle.load(handle)

    standard_samples = sample_for_observation(std_theta_posterior, x_o, n_post_samples=1_000_000)
    twostep_samples = two_step_sampling_from_obs(twostep_z_posterior, twostep_theta_posterior, x_o, 100, 100)







