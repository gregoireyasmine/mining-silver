from .sliced_wasserstein import sliced_wasserstein_distance
from .snpe_utils import sample_for_observation
from .two_step_utils import two_step_sampling_from_obs
from sbi.utils.metrics import c2st
from multiprocessing import Process
import subprocess

def sample_from_estimator(estimator, obs, n_samples):
    if len(estimator)==1:
        return sample_for_observation(estimator[0], obs, n_samples)
    else:
        _, theta_samples = two_step_sampling_from_obs(estimator[0], estimator[1], obs, n_samples, 1)
        # NOTE : by default we take only one theta samples per sampled latent theta which should return more homogeneous samples
        return theta_samples


def dist_for_obs(estimator1, estimator2, method, obs, num_samples, num_projections=50, n_folds=5):
    theta_samples_1 = sample_from_estimator(estimator1, obs, num_samples)
    theta_samples_2 = sample_from_estimator(estimator2, obs, num_samples)
    if method == 'c2st':
        return c2st(theta_samples_1, theta_samples_2, n_folds=n_folds)
    if method == 'wasserstein':
        return sliced_wasserstein_distance(theta_samples_1, theta_samples_2, num_projections=num_projections)
    raise NotImplementedError


def pe_distance(estimator1: tuple, estimator2: tuple, method, observations,
                num_samples=5000, num_projections=50, p=None, n_folds=5):
    """
    Evaluates the two posteriors for a given number of observations, sample and compute
    an empirical metric of similarity between distribution
    Args:
        estimator1 : Tuple : (latent_variables_estimator, theta_estimator) or (theta_direct_estimator, )
        estimator2 : Tuple (latent_variables_estimator, theta_estimator) or (theta_direct_estimator, )
        observations (Tensor): Tensor of x_o observations to evaluate posteriors on
        num_samples: number of thetas sampled from each evaluated distribution
        method (str) : either 'c2st' or 'wasserstein'
        num_projections (int): number of projections to approximate sliced wasserstein distance
        p (int): power of wasserstein distance metric (default 2)
        n_folds (int): number of folds for c2st N-fold cross-validation (default 5)
        max_parallel: how many evaluations to run in parallel
    Return:
        Tensor of size (len(observations)) returning the distance metric for each evaluation of the posteriors
    """
    distances = []
    for obs in observations:
        dist = dist_for_obs(estimator1, estimator2, method, obs, num_samples, num_projections, n_folds)
        distances.append(dist)
    return distances



