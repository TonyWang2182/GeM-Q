"""Clip-average and generator-balanced squared-error objectives."""

import math

import torch


def _vector(value, name):
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")


def _distribution(q):
    _vector(q, "q")
    if not q.is_floating_point() or (q < 0).any():
        raise ValueError("q must contain nonnegative floating-point values")
    if not torch.isclose(q.sum(), q.new_tensor(1.0)):
        raise ValueError("q must sum to one")


def training_counts(groups):
    """Count contiguous generator IDs in the full training partition."""
    _vector(groups, "groups")
    if groups.dtype != torch.long or (groups < 0).any():
        raise ValueError("groups must be nonnegative torch.long IDs")
    counts = torch.bincount(groups)
    if (counts == 0).any():
        raise ValueError("generator IDs must cover 0 through G - 1")
    return counts


def per_clip_weights(counts, group_weights):
    """Return N * w_g / n_g for each generator; index by clip group IDs."""
    _vector(counts, "counts")
    _distribution(group_weights)
    if counts.shape != group_weights.shape or (counts <= 0).any():
        raise ValueError("counts must be positive and match group_weights")
    counts = counts.to(group_weights)
    return counts.sum() * group_weights / counts


def gb_weights(counts):
    """Compute fixed per-clip weights from full-training generator counts."""
    _vector(counts, "counts")
    uniform = torch.full(
        counts.shape, 1.0 / counts.numel(), dtype=torch.float64, device=counts.device
    )
    return per_clip_weights(counts, uniform)


def sgb_weights(counts, q, lam=0.5):
    """Mix uniform and adaptive generator weights, then convert to clip weights."""
    _distribution(q)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must lie in [0, 1]")
    mixture = (1.0 - lam) / q.numel() + lam * q
    return per_clip_weights(counts, mixture)


def erm_mse(prediction, target):
    """Average squared error over clips."""
    _vector(prediction, "prediction")
    if target.shape != prediction.shape:
        raise ValueError("prediction and target must have the same shape")
    return (prediction - target).square().mean()


def weighted_mse(prediction, target, weights):
    """Average weighted squared errors using the batch size as denominator."""
    _vector(prediction, "prediction")
    if target.shape != prediction.shape or weights.shape != prediction.shape:
        raise ValueError("prediction, target, and weights must have the same shape")
    return ((prediction - target).square() * weights.to(prediction)).mean()


@torch.no_grad()
def update_q(q, group_losses, eta=0.1):
    """Update q from full-training group MSEs computed in model.eval() mode."""
    _distribution(q)
    _vector(group_losses, "group_losses")
    if group_losses.shape != q.shape:
        raise ValueError("group_losses must have the same shape as q")
    if not math.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative")
    logq = q.log() + eta * group_losses.detach().to(q)
    logq = logq - logq.max()
    updated = logq.exp()
    return updated / updated.sum()
