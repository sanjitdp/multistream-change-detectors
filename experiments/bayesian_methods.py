"""Bayesian posterior updates and multistream selectors; see README.md."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeometricChangePrior:
    """P(xi=t)=p*h*(1-h)**(t-1), P(xi=infinity)=1-p, for t >= 1."""

    finite_probability: float
    hazard: float

    def __post_init__(self):
        if not 0 <= self.finite_probability <= 1:
            raise ValueError("finite_probability must lie in [0, 1]")
        if not 0 < self.hazard < 1:
            raise ValueError("hazard must lie strictly between 0 and 1")

    def pmf(self, t):
        t = np.asarray(t)
        if np.any(t < 1) or np.any(t != np.floor(t)):
            raise ValueError("pmf time must be an integer >= 1")
        return (
            self.finite_probability
            * self.hazard
            * np.exp((t - 1) * np.log1p(-self.hazard))
        )

    def survival(self, t):
        t = np.asarray(t)
        if np.any(t < 0) or np.any(t != np.floor(t)):
            raise ValueError("survival time must be an integer >= 0")
        return (
            1
            - self.finite_probability
            + self.finite_probability * np.exp(t * np.log1p(-self.hazard))
        )

    def terms(self, t):
        """Return (P(xi=t), P(xi>t-1), P(xi>t)) for either update."""
        return self.pmf(t), self.survival(np.asarray(t) - 1), self.survival(t)


def _check_prior(pmf, previous_survival, survival):
    p, old_s, s = np.broadcast_arrays(
        np.asarray(pmf, float),
        np.asarray(previous_survival, float),
        np.asarray(survival, float),
    )
    if (
        np.any(~np.isfinite(p))
        or np.any(~np.isfinite(old_s))
        or np.any(~np.isfinite(s))
        or np.any(p < 0)
        or np.any(s <= 0)
        or np.any(old_s > 1 + 1e-12)
        or not np.allclose(old_s, s + p, rtol=1e-10, atol=1e-14)
    ):
        raise ValueError("Require positive survival and s(t-1)=s(t)+pmf(t)")
    return p, old_s, s


def posterior_odds_update(
    odds, likelihood_ratio, pmf, previous_survival, survival, active=None
):
    """Update G_t = L_t * (s_(t-1) G_(t-1) + pi_t) / s_t on active streams."""
    p, old_s, s = _check_prior(pmf, previous_survival, survival)
    odds, likelihood_ratio = np.broadcast_arrays(
        np.asarray(odds, float), np.asarray(likelihood_ratio, float)
    )
    if (
        np.any(odds < 0)
        or np.any(np.isnan(odds))
        or np.any(likelihood_ratio < 0)
        or np.any(np.isnan(likelihood_ratio))
    ):
        raise ValueError("Odds and likelihood ratios must be nonnegative")
    updated = likelihood_ratio * (old_s * odds + p) / s
    return updated if active is None else np.where(active, updated, odds)


def log_posterior_odds_update(
    log_odds, log_likelihood_ratio, pmf, previous_survival, survival, active=None
):
    """Update log posterior odds, initialized at -inf, on active streams."""
    p, old_s, s = _check_prior(pmf, previous_survival, survival)
    log_odds, log_likelihood_ratio = np.broadcast_arrays(
        np.asarray(log_odds, float), np.asarray(log_likelihood_ratio, float)
    )
    if np.any(np.isnan(log_odds)) or np.any(np.isnan(log_likelihood_ratio)):
        raise ValueError("Log odds and log likelihood ratios cannot be NaN")
    with np.errstate(divide="ignore"):
        updated = (
            np.logaddexp(log_odds + np.log(old_s), np.log(p))
            + log_likelihood_ratio
            - np.log(s)
        )
    return updated if active is None else np.where(active, updated, log_odds)


def _selection_inputs(values, active, q):
    values = np.asarray(values, float)
    active = np.asarray(active, bool)
    if values.ndim < 1 or values.shape != active.shape or values.shape[-1] < 1:
        raise ValueError("Values and active must have matching (..., K) shapes")
    if not 0 < q < 1 or np.any(np.isnan(values)):
        raise ValueError("Require 0 < q < 1 and non-NaN values")
    return values, active


def _take_prefix(order, r, active):
    selected = np.zeros_like(active)
    rank = np.arange(1, active.shape[-1] + 1)
    np.put_along_axis(selected, order, rank <= r[..., None], axis=-1)
    return selected & active


def md_fdr_rejections(odds, active, q, *, log_input=False):
    """Select the next MD-FDR batch using the original family size K."""
    values, active = _selection_inputs(odds, active, q)
    if not log_input and np.any(values < 0):
        raise ValueError("Posterior odds cannot be negative")
    k = values.shape[-1]
    ranks = np.arange(1, k + 1)
    a = active.sum(axis=-1)
    order = np.argsort(np.where(active, values, -np.inf), axis=-1)[..., ::-1]
    ordered = np.take_along_axis(values, order, axis=-1)
    thresholds = k / (q * (k - a[..., None] + ranks)) - 1
    if log_input:

        thresholds = np.log(np.maximum(thresholds, np.finfo(float).tiny))
    accepted = (ordered >= thresholds) & (ranks <= a[..., None])
    r = np.max(np.where(accepted, ranks, 0), axis=-1)
    return _take_prefix(order, r, active)


def lfdr_rejections(odds, active, q, *, log_input=False):
    """Select the largest prefix whose mean posterior null probability is <= q."""
    values, active = _selection_inputs(odds, active, q)
    if not log_input and np.any(values < 0):
        raise ValueError("Posterior odds cannot be negative")
    if log_input:
        z = np.exp(-np.logaddexp(0.0, values))
    else:
        z = 1 / (1 + values)
    k = values.shape[-1]
    ranks = np.arange(1, k + 1)
    order = np.argsort(np.where(active, z, np.inf), axis=-1)
    ordered = np.take_along_axis(z, order, axis=-1)
    accepted = (np.cumsum(ordered, axis=-1) / ranks <= q) & (
        ranks <= active.sum(axis=-1)[..., None]
    )
    r = np.max(np.where(accepted, ranks, 0), axis=-1)
    return _take_prefix(order, r, active)
