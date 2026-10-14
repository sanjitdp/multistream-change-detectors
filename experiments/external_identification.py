"""Gaussian first-report comparisons from external_identification_design.txt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.special import ndtr

try:
    from .bayesian_methods import (
        GeometricChangePrior,
        lfdr_rejections,
        log_posterior_odds_update,
        md_fdr_rejections,
    )
except ImportError:
    from bayesian_methods import (
        GeometricChangePrior,
        lfdr_rejections,
        log_posterior_odds_update,
        md_fdr_rejections,
    )

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
DESIGN = ROOT / "external_identification_design.txt"
K, H, MU, Q, STEP, CAP = 100, 400, 0.5, 0.05, 0.025, 20.0
ALPHA = Q / H
METHODS = ("BH", "Bonf", "GL", "GL_time", "MD", "LFDR")
LABELS = (
    "e-d-BH (ours)",
    "e-d-Bonferroni (ours)",
    "Gandy-Lau (fixed-time)",
    "Gandy-Lau (time-adjusted)",
    "MD-FDR",
    "Lu LFDR",
)
CASES = (
    ("null", 0, H + 1),
    ("m10_n101", 10, 101),
    ("m50_n101", 50, 101),
    ("m10_n251", 10, 251),
    ("m50_n251", 50, 251),
    ("prior", -1, -1),
)
RANKS = np.arange(1, K + 1)


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def estimate(x):
    x = np.asarray(x, float)
    return (
        (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))))
        if len(x) > 1
        else ((float(x[0]), float("nan")) if len(x) else (float("nan"), float("nan")))
    )


def bh(p, q):
    """Ordinary BH, with inclusive tails and full rejection of threshold ties."""
    ranks = np.arange(1, p.shape[-1] + 1)
    ordered = np.sort(p, axis=-1)
    r = np.max(np.where(ordered <= q * ranks / p.shape[-1], ranks, 0), axis=-1)
    return p <= (q * r / p.shape[-1])[..., None]


def e_bh(log_e, alpha):
    ranks = np.arange(1, log_e.shape[-1] + 1)
    threshold = np.log(log_e.shape[-1] / alpha) - np.log(ranks)
    ordered = np.sort(log_e, axis=-1)[..., ::-1]
    r = np.max(np.where(ordered >= threshold, ranks, 0), axis=-1)
    with np.errstate(divide="ignore"):
        cutoff = np.log(log_e.shape[-1] / alpha) - np.log(r)
    return (log_e >= cutoff[..., None]) & (r[..., None] > 0)


def chain_distribution():
    states = np.arange(round(CAP / STEP) + 1) * STEP
    edges = (np.arange(len(states) - 1) + 0.5) * STEP
    z = (edges[None, :] - states[:, None] + MU**2 / 2) / MU
    transition = np.empty((len(states), len(states)))
    transition[:, 0] = ndtr(z[:, 0])
    transition[:, -1] = ndtr(-z[:, -1])

    transition[:, 1:-1] = np.where(
        z[:, :-1] >= 0,
        ndtr(-z[:, :-1]) - ndtr(-z[:, 1:]),
        ndtr(z[:, 1:]) - ndtr(z[:, :-1]),
    )
    assert np.min(transition) >= 0
    assert np.max(np.abs(transition.sum(axis=1) - 1)) < 5e-15
    mass = np.zeros((H + 1, len(states)))
    mass[0, 0] = 1.0
    for t in range(1, H + 1):
        mass[t] = mass[t - 1] @ transition
    assert np.max(np.abs(mass.sum(axis=1) - 1)) < 1e-12
    tails = np.cumsum(mass[:, ::-1], axis=1)[:, ::-1]
    tails[:, 0] = 1.0
    assert tails[H, -1] < Q / (K * H), "Cap must resolve the smallest BH threshold"
    return transition, mass, tails


def validate(mass, tails, n=100000):
    """Independent chart simulation and direct-CDF recursion check."""
    states = np.arange(mass.shape[1]) * STEP
    edges = (np.arange(mass.shape[1] - 1) + 0.5) * STEP
    z = (edges[None, :] - states[:, None] + MU**2 / 2) / MU
    direct = np.zeros_like(states)
    direct[0] = 1.0
    recursion_error = 0.0
    conditional_cdf = ndtr(z)
    for t in range(1, H + 1):
        cdf = direct @ conditional_cdf
        direct = np.diff(np.r_[0.0, cdf, 1.0])
        recursion_error = max(
            recursion_error,
            float(np.max(np.abs(np.cumsum(direct) - np.cumsum(mass[t])))),
        )
    assert recursion_error < 2e-12

    max_excess = 0.0
    for t in range(1, H + 1):
        for level in (Q, Q / H):
            for cutoff in level * RANKS / K:
                probability = mass[t, tails[t] <= cutoff].sum()
                max_excess = max(max_excess, float(probability - cutoff))
    assert max_excess < 2e-13
    rng = np.random.default_rng(62116999)
    state = np.zeros(n, np.int16)
    rows = []
    for t in range(1, H + 1):
        z = MU * rng.standard_normal(n) - MU**2 / 2
        state = np.floor(np.clip(state * STEP + z, 0, CAP) / STEP + 0.5).astype(
            np.int16
        )
        if t in (1, 20, 100, 200, 400):
            empirical = np.bincount(state, minlength=len(states)) / n
            d = float(np.max(np.abs(np.cumsum(empirical) - np.cumsum(mass[t]))))

            limit = float(np.sqrt(np.log(2 * 5 / 0.001) / (2 * n)))
            assert d < limit, (t, d, limit)
            for cutoff in (0.05, 0.01, 0.001, 0.0005, 0.000125, 0.00000125):
                indicator = tails[t, state] <= cutoff
                observed, se = estimate(indicator)
                rows.append(
                    dict(
                        t=t,
                        n=n,
                        p_cutoff=cutoff,
                        exact_probability=float(mass[t, tails[t] <= cutoff].sum()),
                        empirical_probability=observed,
                        mc_se=se,
                        cdf_sup_error=d,
                        dkw_simultaneous_limit=limit,
                    )
                )

    p = rng.choice(np.r_[0.0, Q * RANKS / K, 1.0], size=(200, K))
    log_e = rng.choice(
        np.r_[-np.inf, np.log(K / ALPHA) - np.log(RANKS), np.inf], size=(200, K)
    )
    for values, got, is_e in ((p, bh(p, Q), False), (log_e, e_bh(log_e, ALPHA), True)):
        for row, actual in zip(values, got):
            ordered = sorted(row, reverse=is_e)
            good = [
                j
                for j, value in enumerate(ordered, 1)
                if (
                    value >= np.log(K / ALPHA) - np.log(j)
                    if is_e
                    else value <= Q * j / K
                )
            ]
            r = max(good, default=0)
            expected = (
                np.zeros(K, bool)
                if r == 0
                else (
                    row >= np.log(K / ALPHA) - np.log(r) if is_e else row <= Q * r / K
                )
            )
            assert np.array_equal(actual, expected)
    write_csv("external_chain_validation.csv", rows)
    report = dict(
        n_null_charts=n,
        seed=62116999,
        recursive_cdf_error=recursion_error,
        exact_superuniform_excess=max_excess,
        selector_cases=400,
        max_empirical_cdf_error=max(r["cdf_sup_error"] for r in rows),
        simultaneous_dkw_limit=rows[0]["dkw_simultaneous_limit"],
        passed=True,
    )
    (OUT / "external_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Validation:", json.dumps(report), flush=True)


def draw_changepoints(rng, n, m, nu):
    xi = np.full((n, K), np.iinfo(np.int32).max, dtype=np.int32)
    if m >= 0:
        xi[:, :m] = nu
    else:
        finite = rng.random((n, K)) < 0.5
        geom = rng.geometric(0.01, size=(n, K))
        xi[finite] = geom[finite]
    return xi


def simulate(case, m, nu, index, n, tails):
    seed = 62106000 + index
    rng = np.random.default_rng(seed)
    xi = draw_changepoints(rng, n, m, nu)
    log_sr = np.full((n, K), -np.inf)
    state = np.zeros((n, K), np.int16)
    log_odds = np.full((n, K), -np.inf)
    prior = GeometricChangePrior(0.5, 0.01)
    first_time = np.full((n, len(METHODS)), H + 1, dtype=np.int16)
    first_true = np.zeros_like(first_time)
    first_false = np.zeros_like(first_time)
    first_changed = np.zeros_like(first_time)
    curves = []
    started = time.time()
    for t in range(1, H + 1):
        changed = xi <= t
        x = rng.standard_normal((n, K)) + MU * changed
        log_l = MU * x - MU**2 / 2
        log_sr = np.logaddexp(0.0, log_sr) + log_l
        state = np.floor(np.clip(state * STEP + log_l, 0, CAP) / STEP + 0.5).astype(
            np.int16
        )
        log_odds = log_posterior_odds_update(log_odds, log_l, *prior.terms(t))
        p = tails[t, state]
        declarations = [
            e_bh(log_sr, ALPHA),
            log_sr >= np.log(K / ALPHA),
            bh(p, Q),
            bh(p, Q / H),
        ]
        for j, selector in enumerate((md_fdr_rejections, lfdr_rejections), 4):
            pending = first_time[:, j] > H
            selected = np.zeros((n, K), bool)
            if np.any(pending):
                values = log_odds[pending]
                selected[pending] = selector(
                    values, np.ones_like(values, bool), Q, log_input=True
                )
            declarations.append(selected)
        for j, selected in enumerate(declarations):
            false = (selected & ~changed).sum(axis=1)
            true = (selected & changed).sum(axis=1)
            count = true + false
            newly = (first_time[:, j] > H) & (count > 0)
            first_time[newly, j] = t
            first_true[newly, j] = true[newly]
            first_false[newly, j] = false[newly]
            first_changed[newly, j] = changed.sum(axis=1)[newly]
            if j < 4 and (t == 1 or t % 5 == 0):
                record = dict(case=case, method=METHODS[j], n=n, t=t)
                for metric, values in (
                    ("fdr", false / np.maximum(1, count)),
                    ("tdr", true / np.maximum(1, changed.sum(axis=1))),
                    ("true_count", true),
                    ("report_size", count),
                ):
                    record[metric], record[metric + "_se"] = estimate(values)
                curves.append(record)
        if t % 100 == 0:
            print(f"{case}: t={t}, elapsed={time.time()-started:.1f}s", flush=True)
    np.savez_compressed(
        OUT / f"external_raw_{case}.npz",
        xi=xi,
        first_time=first_time,
        first_true=first_true,
        first_false=first_false,
        first_changed=first_changed,
        methods=np.array(METHODS),
        seed=seed,
    )
    return summarize_case(case), curves


def summarize_case(case):
    with np.load(OUT / f"external_raw_{case}.npz") as raw:
        xi = raw["xi"]
        onset = xi.min(axis=1)
        times, true, false, changed = (
            raw[key]
            for key in ("first_time", "first_true", "first_false", "first_changed")
        )
        n = len(xi)
        rows = []
        for j, method in enumerate(METHODS):
            report = times[:, j] <= H
            prechange = report & (times[:, j] < onset)
            count = true[:, j] + false[:, j]
            row = dict(case=case, method=method, n=n)
            metrics = dict(
                fdr=false[:, j] / np.maximum(1, count),
                pfer=false[:, j],
                true_count=true[:, j],
                tdr=true[:, j] / np.maximum(1, changed[:, j]),
                report_size=count,
                any_false=false[:, j] > 0,
                no_report=~report,
                prechange_report=prechange,
                report_time=np.minimum(times[:, j], H),
                changed_by_h=(xi <= H).sum(axis=1),
            )
            for metric, values in metrics.items():
                row[metric], row[metric + "_se"] = estimate(values)
            eligible = (onset <= H) & ~prechange
            delay = np.minimum(times[:, j], H) - onset + 1
            observed = eligible & report
            row["delay_n"] = int(eligible.sum())
            row["restricted_delay"], row["restricted_delay_se"] = estimate(
                delay[eligible]
            )
            row["observed_delay_n"] = int(observed.sum())
            row["observed_delay"], row["observed_delay_se"] = estimate(delay[observed])
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.summarize_only:
        rows = [row for case, _, _ in CASES for row in summarize_case(case)]
        write_csv("external_identification_summary.csv", rows)
        return
    started = time.time()
    transition, mass, tails = chain_distribution()
    np.savez_compressed(
        OUT / "external_null_chain.npz", transition=transition, mass=mass, tails=tails
    )
    validate(mass, tails)
    if args.validate_only:
        return
    source_files = (__file__, ROOT / "bayesian_methods.py", DESIGN)
    metadata = dict(
        k=K,
        horizon=H,
        mu=MU,
        q=Q,
        alpha=ALPHA,
        n=args.n,
        step=STEP,
        cap=CAP,
        methods=METHODS,
        labels=LABELS,
        cases=CASES,
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        sha256={
            Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in source_files
        },
    )
    (OUT / "external_identification_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    summary, curves = [], []
    for i, (case, m, nu) in enumerate(CASES):
        s, c = simulate(case, m, nu, i, args.n, tails)
        summary.extend(s)
        curves.extend(c)
        write_csv("external_identification_summary.csv", summary)
        write_csv("external_identification_curves.csv", curves)
        print(f"Completed {case}; total elapsed={time.time()-started:.1f}s", flush=True)
    metadata["elapsed_seconds"] = time.time() - started
    (OUT / "external_identification_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
