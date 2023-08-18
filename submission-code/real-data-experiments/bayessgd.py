"""This library implements the methods described in the paper "Closed-Form Bounds for DP-SGD against Record-level Inference".
It enables:
- computing bounds on the Bayes security of DP-SGD against MIA (both substitution/add-remove relationship)
- instrumenting Opacus to support a data-dependent analysis of DP-SGD against MIA and AI.
"""
import torch
import numpy as np
from scipy.special import erf
from scipy.stats import binom

def point_set_diameter(S, norm=torch.linalg.norm):
    """Returns the max distance (`norm`) between pairwise points in `S`.
    """
    diameter = torch.as_tensor(0.)
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            diameter = torch.maximum(diameter, norm(S[i]-S[j]))

    return diameter

def approximate_point_set_diameter(S, norm=torch.linalg.norm):
    """Returns an upper bound on the max distance (`norm`) between pairwise points in `S`.
    """
    centroid = torch.mean(S, axis=0)
    diameter = torch.as_tensor(0.)
    for x in S:
        diameter = torch.maximum(diameter, norm(x-centroid))

    return torch.as_tensor(2.)*diameter

class AIAnalysis:
    """Data-dependent Bayes security analysis for AI."""
    def __init__(self, attribute_idx, attribute_range, sample_rate, max_grad_norm, noise_multiplier, approximate=False):
        self.attribute_idx = attribute_idx
        self.attribute_range = attribute_range
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.approximate = approximate
        self.sample_rate = sample_rate
        self.max_diff_norms = []

    def _get_gradient_bound(self, x, y, model, optimizer, criterion, device):
        """Augments training record `x` with all the possible completions `self.attribute_range`
        for the attribute at index `self.attribute_idx`, and returns the maximal distance
        between gradients of the augmented data points.
        """
        # Augment `x`'s sensitive attribute with all the possible completions.
        target = y.repeat(len(self.attribute_range))
        # Note: we remove the first dimension of x (which is just a batch indicator).
        augmented = x.repeat(len(self.attribute_range), *[1]*len(x.shape))
        # Select all from the first dimension (i.e., each augmentation); of those,
        # pick the selected attribute and change it to what we need it to be.
        slice_aug = tuple([slice(None, None)] + self.attribute_idx)
        augmented[slice_aug] = torch.as_tensor(self.attribute_range)

        # Compute the gradients for all the augmentations.
        augmented, target = augmented.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(augmented)
        loss = criterion(output, target)
        loss.backward()

        # All gradients (flattened), one row per sample.
        # NOTE: we also need to clip them, because the optimizer (and clipper) hasn't been called
        # on the outputs.
        per_sample_gradients = torch.hstack([p.grad_sample.flatten(1) for p in model.parameters()])
        clip = torch.maximum(torch.as_tensor(1.), torch.linalg.norm(per_sample_gradients, axis=1)/self.max_grad_norm)
        per_sample_gradients = per_sample_gradients / clip.reshape(-1, 1)

        # Let's undo all we've done so far.
        optimizer.zero_grad()

        # Find the distance of the two maximally distant gradients.
        if self.approximate:
            max_diff_norm = approximate_point_set_diameter(per_sample_gradients)
        else:
            max_diff_norm = point_set_diameter(per_sample_gradients)

        return max_diff_norm

    def step(self, model, data, target, optimizer, criterion, device):
        """Should be run for every step. It finds the maximal distance between the gradients
        of two points obtained by replacing their sensitive attribute with its possible values.
        """
        # TODO: it'd be much nicer if we could make this a callback for
        #   optimizer.attach_step_hook()
        # What prevents us from making this change is that we need access to the current batch (data),
        # which isn't given by optimizer.
        # Find bound on gradients difference.
        max_diff_norm = torch.tensor(0.)
        for x, y in zip(data, target):
            max_diff_norm = torch.maximum(max_diff_norm, self._get_gradient_bound(x, y, model, optimizer, criterion, device))

        self.max_diff_norms.append(max_diff_norm.to("cpu"))

    @property
    def beta(self):
        return 1-erf(self.sample_rate*np.linalg.norm(self.max_diff_norms)/(2*np.sqrt(2)*self.noise_multiplier*self.max_grad_norm))


class MIAAnalysis:
    """Data-independent Bayes-SGD MIA analysis.
    """
    def __init__(self, noise_multiplier, sample_rate, approximate=False):
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.approximate = approximate
        self.steps = 0

    def step(self):
        """Must be called at the end of each step to update the
        Bayes security value.
        """
        self.steps += 1

    @property
    def beta(self):
        return 1-erf(self.sample_rate*np.sqrt(self.steps)/(np.sqrt(2)*self.noise_multiplier))