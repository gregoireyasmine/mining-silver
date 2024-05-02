import torch
from sbi.inference import SNPE, simulate_for_sbi
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from torch import Tensor
from torch.distributions import Distribution, Uniform
from .snpe_utils import sample_for_observation
from typing import Callable, List, Optional


def simulate_two_step(latent_variables_simulator: Callable,
                      from_latent_simulator: Callable,
                      theta_prior: Distribution,
                      num_simul: int):
    """
    Processes prior and simulator, samples sets of parameters from the priors and returns simulated latent variables and
    corresponding outputs.

    Args:
        latent_variables_simulator: A simulator generating latent variables given a set of parameters
        from_latent_simulator: A simulator generating output variables given a set of latent variables
        theta_prior: The prior distribution in Pytorch format.
        num_simul: The number of simulations to run
    Returns:
        theta_samples: Samples of parameters from the prior
        lv_samples: Samples of latent variables generated given theta_samples
        x_samples : Samples of x variables generated given lv_samples
    """
    prior, num_parameters, prior_returns_numpy = process_prior(theta_prior)
    lv_simulator = process_simulator(latent_variables_simulator, prior, prior_returns_numpy)
    check_sbi_inputs(lv_simulator, theta_prior)

    theta_samples = prior.sample((num_simul,))
    lv_samples = lv_simulator(theta_samples)
    lv_prior = BoxUniform(low=lv_samples.min(0)[0], high=lv_samples.max(0)[0])  # We won't use this prior, it's just for preprocessing
    lv_prior, lv_dim, lv_prior_returns_numpy = process_prior(lv_prior)
    from_lv_simulator = process_simulator(from_latent_simulator, lv_prior, prior_returns_numpy)
    check_sbi_inputs(from_lv_simulator, theta_prior)

    x_samples = from_lv_simulator(lv_samples)

    return theta_samples, lv_samples, x_samples


def two_step_sampling_from_obs(lv_posterior: Distribution, theta_posterior: Distribution, x_obs: Tensor,
                               num_lv_samples: int, num_samples_per_latent: int):
    """
    Draws sample from latent variables posterior distribution evaluated at x_obs, then for each sample z, samples from
    posterior distribution of parameters evaluated at (z, x_obs)

    Args:
        lv_posterior: The posterior of latent variables conditioned to output
        theta_posterior: The posterior of parameters conditioned to latent variables and output
        x_obs: The observation at which the posterior has to be evaluated
        num_lv_samples: How many samples of latent variables have to be drawn
        num_samples_per_latent: How many samples of theta have to be drawn for each sample of latent variables
    """
    x_obs = x_obs.squeeze()
    lv_posterior_samples = sample_for_observation(lv_posterior, x_obs, num_lv_samples)
    theta_posterior_samples = []
    for z_p in lv_posterior_samples:
        lv_x_obs = torch.concatenate([z_p, x_obs])
        theta_posterior_samples.append(sample_for_observation(theta_posterior, lv_x_obs, n_post_samples=num_samples_per_latent))
    theta_posterior_samples = torch.concatenate(theta_posterior_samples)
    return lv_posterior_samples, theta_posterior_samples


