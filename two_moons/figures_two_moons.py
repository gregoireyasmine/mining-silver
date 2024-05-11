import matplotlib.pyplot as plt
import pickle
from utils.snpe_utils import sample_for_observation
from utils.two_step_utils import two_step_sampling_from_obs, simulate_two_step
from two_moons import from_means, mean_function
from sbi.utils import BoxUniform
import torch
import os
import numpy as np
from sbi import analysis as analysis

SIM_BUDGETS = [100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000, 20000]
NUM_OBS = 10
NUM_SAMPLES = 500
ROOT = os.getcwd() + '/..'
RESULTS_DIR = os.path.join(ROOT, 'results/mean_distances')
MODEL_DIR = os.path.join(ROOT, 'validation/two_moons')
FIG_DIR = os.path.join(ROOT, 'figures')
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)


## Figure 1: evaluate distances between two methods over simulation budgets, or within a method

# c2st_between_methods_file = os.path.join(RESULTS_DIR)
# with open(RESULTS_DIR + f'/mean_c2st_distance_between_methods_{NUM_OBS}obs_{NUM_SAMPLES}samples.pickle', 'rb') as handle:
#    mean_c2st_2_methods = pickle.load(handle)

with open(RESULTS_DIR + f'/mean_wasserstein_distances_between_two_methods_inferers_{10}obs_{10000}samples.pickle', 'rb') as handle:
    mean_wasserstein_2_methods = pickle.load(handle)

mean_wasserstein_2_methods = [np.mean(dist) if len(dist) > 0 else None for dist in mean_wasserstein_2_methods ]


with open(RESULTS_DIR + f'/mean_wasserstein_distances_between_twostep_inferers_{10}obs_{10000}samples.pickle', 'rb') as handle:
    mean_wasserstein_twostep = pickle.load(handle)
print(mean_wasserstein_twostep)
mean_wasserstein_twostep = [np.mean(dist) if len(dist) > 0 else None for dist in mean_wasserstein_twostep ]


with open(RESULTS_DIR + f'/mean_wasserstein_distances_between_standard_inferers_{10}obs_{10000}samples.pickle', 'rb') as handle:
    mean_wasserstein_std = pickle.load(handle)
print(mean_wasserstein_std)
mean_wasserstein_std = [np.mean(dist) if len(dist) > 0 else None for dist in mean_wasserstein_std ]
fig, ax = plt.subplots(1, 2, figsize=(8, 6), sharex=True)

ax[0].semilogx(SIM_BUDGETS[:-1], mean_wasserstein_2_methods[:-1])
ax[0].set_title('Sliced Wasserstein distance')
ax[0].set_xlabel('Number of simulations')
ax[0].set_ylabel('Wasserstein distance')

ax[1].semilogx(SIM_BUDGETS[:-1], mean_wasserstein_std[:-1])
ax[1].set_title('Sliced Wasserstein distance')
ax[1].set_xlabel('Number of simulations')
ax[1].set_ylabel('Wasserstein distance')

ax[1].semilogx(SIM_BUDGETS[:-1], mean_wasserstein_twostep[:-1])
ax[1].set_title('Sliced Wasserstein distance')
ax[1].set_xlabel('Number of simulations')
ax[1].set_ylabel('Wasserstein distance')


fig.suptitle('Distance between posteriors, averaged over observations')

plt.tight_layout()
plt.savefig(FIG_DIR + f'/wass_distance_between_methods_{10}_obs_{10000}_samples')


with open(RESULTS_DIR + f'/mean_c2st_distances_between_two_methods_inferers_{10}obs_{10000}samples.pickle', 'rb') as handle:
    mean_c2st_2_methods = pickle.load(handle)

mean_c2st_2_methods = [np.mean(dist) if len(dist) > 0 else None for dist in mean_c2st_2_methods ]
print(mean_c2st_2_methods)

with open(RESULTS_DIR + f'/mean_c2st_distances_between_twostep_inferers_{10}obs_{500}samples.pickle', 'rb') as handle:
    mean_c2st_twostep = pickle.load(handle)
print(mean_c2st_twostep)
mean_c2st_twostep = [np.mean(dist) if len(dist) > 0 else None for dist in mean_c2st_twostep ]


with open(RESULTS_DIR + f'/mean_c2st_distances_between_standard_inferers_{10}obs_{500}samples.pickle', 'rb') as handle:
    mean_c2st_std = pickle.load(handle)
print(mean_c2st_std)
mean_c2st_std = [np.mean(dist) if len(dist) > 0 else None for dist in mean_c2st_std ]
fig, ax = plt.subplots(1, 2, figsize=(8, 6), sharex=True)

ax[0].semilogx(SIM_BUDGETS[:-1], mean_c2st_2_methods[:-1])
ax[0].set_title('c2st accuracy')
ax[0].set_xlabel('Number of simulations')
ax[0].set_ylabel('c2st accuracy')

ax[1].semilogx(SIM_BUDGETS[:-1], mean_c2st_std[:-1])
ax[1].set_title('c2st accuracy')
ax[1].set_xlabel('Number of simulations')
ax[1].set_ylabel('c2st accuracy')

ax[1].semilogx(SIM_BUDGETS[:-1], mean_c2st_twostep[:-1])
ax[1].set_title('c2st accuracy')
ax[1].set_xlabel('Number of simulations')
ax[1].set_ylabel('c2st accuracy')


fig.suptitle('C2ST accuracy between posteriors, averaged over observations')

plt.tight_layout()
plt.savefig(FIG_DIR + f'/c2st_distance_between_methods_{NUM_OBS}_obs_{NUM_SAMPLES}_samples')

## Figure 2: evaluate distances between method and standard ground truth

with open(RESULTS_DIR + f'/mean_c2st_distances_standard_inferers_to_round_no_1_50000_sim_standard_theta_results_10obs_10000samples.pickle', 'rb') as handle:
    std_std_c2st = pickle.load(handle)
print(std_std_c2st)
std_std_c2st = [np.mean(dist) if len(dist)>0 else None for dist in std_std_c2st]

with open(RESULTS_DIR + f'/mean_c2st_distances_standard_inferers_to_round_no_1_50000_sim_twostep_theta_results_10obs_2000samples.pickle', 'rb') as handle:
    std_twostep_c2st = pickle.load(handle)

std_twostep_c2st = [np.mean(dist) if len(dist)>0 else None for dist in std_twostep_c2st]

with open(RESULTS_DIR + f'/mean_c2st_distances_twostep_inferers_to_round_no_1_50000_sim_standard_theta_results_10obs_2000samples.pickle', 'rb') as handle:
    twostep_std_c2st = pickle.load(handle)

twostep_std_c2st = [np.mean(dist) if len(dist)>0 else None for dist in twostep_std_c2st]

with open(RESULTS_DIR + f'/mean_c2st_distances_twostep_inferers_to_round_no_1_50000_sim_twostep_theta_results_10obs_2000samples.pickle', 'rb') as handle:
    twostep_twostep_c2st = pickle.load(handle)

twostep_twostep_c2st = [np.mean(dist) if len(dist)>0 else None for dist in twostep_twostep_c2st]

with open(RESULTS_DIR + f'/mean_wasserstein_distances_standard_inferers_to_round_no_1_50000_sim_standard_theta_results_10obs_10000samples.pickle', 'rb') as handle:
    std_std_wasserstein = pickle.load(handle)

std_std_wasserstein = [np.mean(dist) if len(dist)>0 else None for dist in std_std_wasserstein]

with open(RESULTS_DIR + f'/mean_wasserstein_distances_standard_inferers_to_round_no_1_50000_sim_twostep_theta_results_10obs_2000samples.pickle', 'rb') as handle:
    std_twostep_wasserstein = pickle.load(handle)

std_twostep_wasserstein = [np.mean(dist) if len(dist)>0 else None for dist in std_twostep_wasserstein]

with open(RESULTS_DIR + f'/mean_wasserstein_distances_twostep_inferers_to_round_no_1_50000_sim_standard_theta_results_10obs_2000samples.pickle', 'rb') as handle:
    twostep_std_wasserstein = pickle.load(handle)

twostep_std_wasserstein = [np.mean(dist) if len(dist)>0 else None for dist in twostep_std_wasserstein]

with open(RESULTS_DIR + f'/mean_wasserstein_distances_twostep_inferers_to_round_no_1_50000_sim_twostep_theta_results_10obs_2000samples.pickle', 'rb') as handle:
    twostep_twostep_wasserstein = pickle.load(handle)

twostep_twostep_wasserstein = [np.mean(dist) if len(dist)>0 else None for dist in twostep_twostep_wasserstein]

fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
ax[0].semilogx(SIM_BUDGETS[:len(twostep_twostep_c2st)], twostep_twostep_c2st, label='two-step NPE')
ax[0].semilogx(SIM_BUDGETS[:len(std_twostep_c2st)], std_twostep_c2st, label='standard NPE')
ax[0].set_title('C2ST distance')
ax[0].legend()
ax[1].semilogx(SIM_BUDGETS[:len(twostep_twostep_wasserstein)], twostep_twostep_wasserstein)
ax[1].semilogx(SIM_BUDGETS[:len(std_twostep_wasserstein)], std_twostep_wasserstein)

ax[1].set_title('Sliced Wasserstein distance')

ax[0].set_xlabel('Number of simulations')
ax[1].set_xlabel('Number of simulations')
ax[0].set_ylabel('C2ST accuracy')
ax[1].set_ylabel('Wasserstein distance')

#ax[0].set_ylim(0.5, 0.99)
fig.suptitle('Distance to ground truth (twostep), averaged over observations')

plt.tight_layout()
plt.savefig(FIG_DIR + f'/distance_to_twostep_limit_{NUM_OBS}_obs_{NUM_SAMPLES}_samples')


fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
ax[0].semilogx(SIM_BUDGETS[:len(twostep_std_c2st)], twostep_std_c2st, label='two-step NPE')
ax[0].semilogx(SIM_BUDGETS[:len(std_std_c2st)], std_std_c2st, label='standard NPE')
ax[0].set_title('C2ST distance')
ax[0].legend()
ax[1].semilogx(SIM_BUDGETS[:len(twostep_std_wasserstein)], twostep_std_wasserstein)
ax[1].semilogx(SIM_BUDGETS[:len(std_std_wasserstein)], std_std_wasserstein)

ax[1].set_title('Sliced Wasserstein distance')

ax[0].set_xlabel('Number of simulations')
ax[1].set_xlabel('Number of simulations')
ax[0].set_ylabel('C2ST accuracy')
ax[1].set_ylabel('Wasserstein distance')

fig.suptitle('Distance to ground truth (standard), averaged over observations')

plt.tight_layout()
plt.savefig(FIG_DIR + f'/distance_to_standard_limit_{NUM_OBS}_obs_{NUM_SAMPLES}_samples')


theta_prior = BoxUniform(low=torch.tensor([-1.0, -1.0]), high=torch.tensor([1.0, 1.0]))
true_theta, true_z, x_o = simulate_two_step(mean_function, from_means, theta_prior, 1)

"""
FIGSIZE = (15, 15)
for n_sim in SIM_BUDGETS:
    for i in range(1, 9):
        try:
            with open(MODEL_DIR+f'/round_no_{i}_{n_sim}_sim_standard_theta_results.pickle', 'rb') as handle:
                _, _, std_theta_posterior = pickle.load(handle)

                standard_samples = sample_for_observation(std_theta_posterior, x_o, n_post_samples=1_000_000)

                plt.figure()
                analysis.pairplot(standard_samples, points=true_theta, limits=[[-1, 1], [-1, 1]], figsize=FIGSIZE,
                                  labels=[r"$\theta_1$", r"$\theta_2$"], upper='kde')
                plt.savefig(os.path.join(FIG_DIR, f'{n_sim}_sim_standard_theta_posterior_plot.png'))
                plt.close()
                
            break
        except FileNotFoundError:
            if i == 8:
                print(MODEL_DIR+f'/round_no_{i}_{n_sim}_sim_standard_theta_results.pickle not Found')

    for i in range(1, 9):
        try:
            with open(MODEL_DIR+'/round_no_1_' + str(n_sim) + '_sim_twostep_z_results.pickle', 'rb') as handle:
                _,_,twostep_z_posterior = pickle.load(handle)
            with open(MODEL_DIR+'/round_no_1_' + str(n_sim) + '_sim_twostep_theta_results.pickle', 'rb') as handle:
                _,_,twostep_theta_posterior = pickle.load(handle)

            z_twostep_samples, theta_twostep_samples = two_step_sampling_from_obs(twostep_z_posterior, twostep_theta_posterior, x_o, 10000, 1000)

            plt.figure()
            analysis.pairplot(theta_twostep_samples, points=true_theta, limits=[[-1, 1], [-1, 1]], figsize=FIGSIZE,
                              labels=[r"$\theta_1$", r"$\theta_2$"])
            plt.savefig(os.path.join(FIG_DIR, f'{n_sim}_sim_twostep_theta_posterior_plot'))
            plt.close()

            plt.figure()
            analysis.pairplot(z_twostep_samples, points=true_theta, limits=[[-1, 1], [-1, 1]], figsize=FIGSIZE,
                              labels=[r"$z_1$", r"$z_2$"])
            plt.savefig(os.path.join(FIG_DIR, f'{n_sim}_sim_twostep_z_posterior_plot'))
            plt.close()
                
            break
        except FileNotFoundError:
            if i == 8:
                print(MODEL_DIR+f'/round_no_{i}_{n_sim}_sim_standard_theta_results.pickle not Found')
"""
'''
with open(MODEL_DIR + '/round_no_1_' + str(50000) + '_sim_twostep_z_results.pickle', 'rb') as handle:
    _, _, twostep_z_posterior = pickle.load(handle)
with open(MODEL_DIR + '/round_no_1_' + str(50000) + '_sim_twostep_theta_results.pickle', 'rb') as handle:
    _, _, twostep_theta_posterior = pickle.load(handle)

z_twostep_samples, theta_twostep_samples = two_step_sampling_from_obs(twostep_z_posterior, twostep_theta_posterior, x_o, 1000, 1000)

plt.figure()
analysis.pairplot(theta_twostep_samples, points=true_theta, limits=[[-1, 1], [-1, 1]], figsize=(10, 10),
                  labels=[r"$\theta_1$", r"$\theta_2$"])
plt.savefig(os.path.join(FIG_DIR, f'{50000}_sim_twostep_theta_posterior_plot'))
plt.close()

plt.figure()
analysis.pairplot(z_twostep_samples, points=true_theta, limits=[[-1, 1], [-1, 1]], figsize=(10, 10),
                  labels=[r"$z_1$", r"$z_2$"])
plt.savefig(os.path.join(FIG_DIR, f'{50000}_sim_twostep_z_posterior_plot'))
plt.close()
'''