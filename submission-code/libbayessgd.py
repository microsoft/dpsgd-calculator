import numpy as np
from scipy import stats
from scipy.special import erf
import fourier_accountant


def compute_beta_mia(p, sigma, T, relation="s"):
    """Corollary 8: compute the Bayes security for MIA, w.r.t. a given relation ("s" or "r").

    Args:
        p (float): sample rate
        sigma (float): noise multiplier
        T (float): number of steps
        relation (str, optional): "s" for substitution relation, "r" for add-remove relation. Defaults to "s".
    """
    if relation == "s":
        return 1-erf(p*np.sqrt(T)/(np.sqrt(2)*sigma))
    elif relation == "r":
        return 1-erf(p*np.sqrt(T)/(2*np.sqrt(2)*sigma))
    else:
        raise ValueError("relation must be either 's' or 'r'")

def tv_error_term(p, sigma, T):
    """Computes the approximation error on the TV (Proposition 4).

    Args:
        p (float): sample rate
        sigma (float): noise multiplier
        T (float): number of steps
        """
    return np.sqrt(T*(p - p**2))/(2*sigma)

#########################################################################
# Accountant methods.
#########################################################################

def compute_beta_pld(p, sigma, T, relation="s"):
    """Computes Bayes security for MIA via https://github.com/DPBayes/PLD-Accountant.

    Args:
        p (float): sample rate
        sigma (float): noise multiplier
        T (float): number of steps
        relation (str, optional): "s" for substitution relation, "r" for add-remove relation. Defaults to "s".
    """
    if relation == "s":
        delta = fourier_accountant.get_delta_S(target_eps=10e-6, sigma=sigma, q=p, ncomp=T)
    elif relation == "r":
        delta = fourier_accountant.get_delta_R(target_eps=10e-6, sigma=sigma, q=p, ncomp=T)
    else:
        raise ValueError("relation must be either 's' or 'r'")
    return 1-delta

def compute_beta_prv(p, sigma, C, T, mode="estimate", relation="s"):
    """Computes Bayes security for MIA via the PRV accountant.

    NOTE: this may be unreliable for the substitution relation.

    Args:
        p (float): sample rate
        sigma (float): noise multiplier
        C (float): clipping bound
        T (float): number of steps
        mode (str, optional): "lower", "upper", or "estimate". Defaults to "estimate".
        relation (str, optional): "s" for substitution relation, "r" for add-remove relation. Defaults to "s".
    """
    T = int(T)
    if relation == "s":
        C *= 2
    acc = PRVAccountant(prvs=PoissonSubsampledGaussianMechanism(
                            noise_multiplier=sigma/C,
                            sampling_probability=p),
                        max_self_compositions=T,
                        eps_error=10**-4,
                        delta_error=10**-4,
        )

    # Note: we use the upper bound on delta.
    delta_lower, delta_est, delta_upper = acc.compute_delta(num_self_compositions=T, epsilon=0)
    if mode == "lower":
        delta = delta_lower
    elif mode == "upper":
        delta = delta_upper
    elif mode == "estimate":
        delta = delta_est
    else:
        raise NotImplementedError

    return 1-delta