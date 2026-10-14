"""Check Bayesian posterior recursions and selection rules."""

import itertools

import numpy as np
from bayesian_methods import (
    GeometricChangePrior,
    lfdr_rejections,
    log_posterior_odds_update,
    md_fdr_rejections,
    posterior_odds_update,
)


def main():
    rng = np.random.default_rng(720061)

    pmf = np.array([0.1, 0.0, 0.3, 0.2, 0.05])
    survival = 1 - np.r_[0.0, np.cumsum(pmf)]
    lr = np.exp(rng.normal(size=(len(pmf), 7, 4)))
    odds = np.zeros((7, 4))
    log_odds = np.full_like(odds, -np.inf)
    max_relative_error = 0.0
    for t in range(1, len(pmf) + 1):
        args = (pmf[t - 1], survival[t - 1], survival[t])
        odds = posterior_odds_update(odds, lr[t - 1], *args)
        log_odds = log_posterior_odds_update(log_odds, np.log(lr[t - 1]), *args)
        direct = sum(pmf[j] * lr[j:t].prod(axis=0) for j in range(t)) / survival[t]
        np.testing.assert_allclose(odds, direct, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(np.exp(log_odds), direct, rtol=1e-13, atol=1e-14)
        max_relative_error = max(max_relative_error, np.max(abs(odds / direct - 1)))

    selectors_checked = 0
    for k in range(1, 8):
        q = 0.2
        values = np.exp(rng.uniform(-4, 7, size=(40, k)))
        active = rng.uniform(size=values.shape) < 0.7
        actual = md_fdr_rejections(values, active, q)
        np.testing.assert_array_equal(
            actual, md_fdr_rejections(np.log(values), active, q, log_input=True)
        )
        for row in range(len(values)):
            ascending = sorted(
                np.flatnonzero(active[row]), key=lambda j: values[row, j]
            )
            expected = np.zeros(k, bool)
            for l, index in enumerate(ascending, start=1):
                if values[row, index] >= k / (q * (k - l + 1)) - 1:
                    expected[ascending[l - 1 :]] = True
                    break
            np.testing.assert_array_equal(actual[row], expected)
            selectors_checked += 1

    np.testing.assert_array_equal(
        md_fdr_rejections(
            np.array([100.0, 100.0, 6.0, 0.1]),
            np.array([False, False, True, True]),
            0.2,
        ),
        np.array([False, False, True, False]),
    )

    subsets_checked = 0
    for k in range(1, 8):
        for _ in range(30):
            z = rng.uniform(0.001, 0.999, k)
            odds = 1 / z - 1
            active = rng.uniform(size=k) < 0.8
            q = 0.2
            actual = lfdr_rejections(odds, active, q)
            np.testing.assert_array_equal(
                actual, lfdr_rejections(np.log(odds), active, q, log_input=True)
            )
            assert not np.any(actual & ~active)
            assert z[actual].sum() / max(actual.sum(), 1) <= q + 1e-13
            best = 0.0
            for bits in itertools.product([False, True], repeat=k):
                candidate = np.asarray(bits) & active
                if z[candidate].sum() / max(candidate.sum(), 1) <= q:
                    best = max(best, np.sum(1 - z[candidate]))
                subsets_checked += 1
            np.testing.assert_allclose(np.sum(1 - z[actual]), best, atol=1e-13)

    tied_odds = np.array([[9.0, 9.0, 9.0], [1.0, 1.0, 1.0]])
    none = np.zeros_like(tied_odds, bool)
    assert not md_fdr_rejections(tied_odds, none, 0.2).any()
    assert not lfdr_rejections(tied_odds, none, 0.2).any()
    np.testing.assert_array_equal(
        lfdr_rejections(tied_odds, ~none, 0.2),
        [[True, True, True], [False, False, False]],
    )
    prior = GeometricChangePrior(0.8, 0.1)
    assert prior.survival(0) == 1
    np.testing.assert_allclose(prior.pmf(np.arange(1, 6)), 0.08 * 0.9 ** np.arange(5))
    log_odds = log_posterior_odds_update(
        np.array([-np.inf, 5.0]),
        np.array([800.0, -800.0]),
        *prior.terms(1),
        active=[True, False],
    )
    assert np.isfinite(log_odds[0]) and log_odds[1] == 5.0
    assert lfdr_rejections(
        log_odds, np.array([True, False]), 0.05, log_input=True
    ).tolist() == [True, False]
    print(
        f"PASS: posterior direct-enumeration relative error {max_relative_error:.2g}; "
        f"{selectors_checked} literal MD-FDR comparisons; "
        f"{subsets_checked} exhaustive LFDR candidate subsets; "
        "log-space, inactivity, carry-over and tie checks."
    )


if __name__ == "__main__":
    main()
