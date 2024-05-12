from torch import Tensor
from sbi.inference import SNPE


def train_inferer(prior_samples: Tensor, outputs: Tensor, design: str = 'nsf'):
    """
    Given prior samples and corresponding simulating output, trains a neural posterior estimator

    Args:
        prior_samples : Tensor of prior samples
        outputs : Tensor of corresponding simulator outputs
        design : The density estimator architecture to use (see SBI documentation).

    Returns :
        (inference, density_estimator, posterior) : results of the posterior estimation process
    """
    inference = SNPE(density_estimator=design)
    inference = inference.append_simulations(prior_samples, outputs)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator=density_estimator)
    return inference, density_estimator, posterior


def sample_for_observation(posterior, obs, n_post_samples):
    posterior.set_default_x(obs)
    post_samples = posterior.sample((n_post_samples,), show_progress_bars=False)
    return post_samples



