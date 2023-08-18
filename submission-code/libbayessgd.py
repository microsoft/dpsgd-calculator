import numpy as np
import fourier_accountant
from scipy.special import erf
from shapely.geometry import Polygon


def compute_beta_mia(p, sigma, T, relation="s"):
    """Corollary 7: compute the Bayes security for MIA, w.r.t. a given relation ("s" or "r").

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

#############################
# TPR-FPR curves from f-DP. #
#############################
def intersect_polygons(polygons):
    if len(polygons) == 1:
        return polygons[0]
    elif len(polygons) == 2:
        return polygons[0].intersection(polygons[1])
    else:
        mid = len(polygons) // 2
        left = intersect_polygons(polygons[:mid])
        right = intersect_polygons(polygons[mid:])
        return left.intersection(right)

def convert_eps_deltas_to_fpr_fnr(epsilons, deltas):
    polygons = [
        Polygon([(0, 1), (0, 1-d), ((1-d)/(1+np.exp(e)), (1-d)/(1+np.exp(e))), (1-d, 0), (1, 0)])
        for e, d in zip(epsilons, deltas)
    ]
    intersection = intersect_polygons(polygons)
    return np.array(intersection.exterior.coords.xy[0]), np.array(intersection.exterior.coords.xy[1])

def pld_tpr_fpr(N, L, sigma, T, deltas=np.logspace(-6, 0, 20, endpoint=False)):
    """Compute TPR and FPR curves using PLD."""
    epsilons = [fourier_accountant.get_epsilon_S(target_delta=delta, sigma=sigma, q=L/N, ncomp=T) for delta in deltas]
    fpr, fnr = convert_eps_deltas_to_fpr_fnr(epsilons, deltas)

    return fpr, 1-fnr