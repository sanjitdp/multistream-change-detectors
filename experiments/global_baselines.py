#!/usr/bin/env python3
"""Xie--Siegmund and Chan global detectors from global_external_design.txt."""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
K = 50
H = 400
CHANGE = 101
TARGET = 1000
W = 199
METHODS = ("Xie--Siegmund T2", "Chan transformed CUSUM")
LIB = None


def library():
    global LIB
    if LIB is None:
        source = ROOT / "global_baselines.cpp"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        mac = platform.system() == "Darwin"
        path = (
            Path(tempfile.gettempdir())
            / f"global_baselines_{digest}{'.dylib' if mac else '.so'}"
        )
        if not path.exists():
            compiler = shutil.which("clang++") or shutil.which("g++")
            if compiler is None:
                raise RuntimeError(
                    "A C++17 compiler is required for the scan statistic."
                )
            flags = (
                ["-dynamiclib", "-framework", "Accelerate"]
                if mac
                else ["-shared", "-fPIC"]
            )
            command = [
                compiler,
                "-O3",
                "-std=c++17",
                *flags,
                str(source),
                "-o",
                str(path),
            ]
            subprocess.run(command, check=True)
        LIB = ctypes.CDLL(str(path))
        LIB.gb_create.argtypes = [ctypes.c_int] * 3
        LIB.gb_create.restype = ctypes.c_void_p
        LIB.gb_destroy.argtypes = [ctypes.c_void_p]
        LIB.gb_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
    return LIB


class Statistics:
    def __init__(self, n, k=K, window=W):
        self.lib = library()
        self.ptr = self.lib.gb_create(n, k, window)
        self.result = np.empty((n, 2), dtype=np.float64)

    def step(self, x, active, t):
        x = np.ascontiguousarray(x, dtype=np.float64)
        active = np.ascontiguousarray(active, dtype=np.uint8)
        self.lib.gb_step(
            self.ptr, x.ctypes.data, active.ctypes.data, t, self.result.ctypes.data
        )
        return self.result

    def close(self):
        if self.ptr:
            self.lib.gb_destroy(self.ptr)
            self.ptr = None


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def estimate(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def checks_and_timing():
    rng = np.random.default_rng(72000)
    x = rng.normal(size=(45, 3, 4))
    s = Statistics(3, 4, 11)
    cumulative = np.concatenate((np.zeros((1, 3, 4)), np.cumsum(x, axis=0)))
    r = np.zeros((3, 4))
    max_error = 0.0
    for t in range(1, 46):
        actual = s.step(x[t - 1], np.full(3, 3, dtype=np.uint8), t).copy()
        lengths = np.arange(1, min(t, 11) + 1)
        u = (cumulative[t][None, :, :] - cumulative[t - lengths]) / np.sqrt(
            lengths[:, None, None]
        )
        expected = (
            np.logaddexp(np.log(0.9), np.log(0.1) + np.maximum(u, 0) ** 2 / 2)
            .sum(axis=2)
            .max(axis=0)
        )
        r = np.maximum(0, r + x[t - 1] - 0.5)
        chan = np.logaddexp(np.log(0.9), np.log(0.064) + r / 2).sum(axis=1)
        max_error = max(
            max_error, float(np.max(np.abs(actual - np.column_stack((expected, chan)))))
        )
    s.close()
    assert max_error < 1e-10, max_error

    s = Statistics(32)
    active = np.full(32, 3, dtype=np.uint8)
    start = time.perf_counter()
    for t in range(1, 501):
        s.step(rng.normal(size=(32, K)), active, t)
    elapsed = time.perf_counter() - start
    s.close()
    result = dict(
        direct_check_max_error=max_error,
        timing_runs=32,
        timing_steps=500,
        elapsed_seconds=elapsed,
        seconds_per_path_step=elapsed / (32 * 500),
        seed=72000,
    )
    (OUT / "global_external_checks.json").write_text(json.dumps(result, indent=2))
    print("Checks and timing:", result, flush=True)
    return result


def xs_initial_bracket():
    """Published asymptotic ARL formula, used only to start training search."""
    from numpy.polynomial.hermite import hermgauss
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from scipy.special import ndtr

    x, w = hermgauss(200)
    u = np.sqrt(2) * x
    w = w / np.sqrt(np.pi)
    z = np.maximum(u, 0)
    g = np.logaddexp(np.log(0.9), np.log(0.1) + z * z / 2)
    derivative = z / (1 + 9 * np.exp(-z * z / 2))

    def evaluate(theta):
        a = np.exp(theta * g)
        ea = w @ a
        psi = np.log(ea)
        tilted = w * a / ea
        mean = tilted @ g
        variance = tilted @ (g * g) - mean * mean
        gamma = 0.5 * theta * theta * (tilted @ (derivative * derivative))
        b = K * mean

        def nu(y):
            return (
                (2 / y)
                * (ndtr(y / 2) - 0.5)
                / ((y / 2) * ndtr(y / 2) + np.exp(-y * y / 8) / np.sqrt(2 * np.pi))
            )

        integral = quad(
            lambda y: y * nu(y) ** 2,
            np.sqrt(2 * K * gamma / 200),
            np.sqrt(2 * K * gamma),
        )[0]
        arl = (
            theta
            * np.sqrt(2 * np.pi * variance)
            / (gamma * np.sqrt(K))
            * np.exp(theta * b - K * psi)
            / integral
        )
        return b, arl

    theta = brentq(lambda q: np.log(evaluate(q)[1] / TARGET), 0.6, 0.8)
    threshold, arl = evaluate(theta)
    return threshold + 0.45, dict(
        approximate_threshold=threshold,
        approximate_arl=arl,
        theta=theta,
        bracket_margin=0.45,
    )


def calibrate(n=400, seed=73100):
    xs_hi, approx = xs_initial_bracket()
    brackets = np.array([xs_hi, 3.0])
    all_attempts = []
    for attempt in range(6):
        rng = np.random.default_rng(seed)
        s = Statistics(n)
        times = np.zeros((n, 2), dtype=np.int32)
        running = np.full((n, 2), -np.inf)
        records = [[[] for _ in range(n)] for _ in range(2)]
        start = time.monotonic()
        for t in range(1, 100001):
            active = (times[:, 0] == 0).astype(np.uint8) + 2 * (
                times[:, 1] == 0
            ).astype(np.uint8)
            stat = s.step(rng.standard_normal((n, K)), active, t)
            for j in range(2):
                new = np.flatnonzero(stat[:, j] > running[:, j])
                for i in new:
                    records[j][i].append((float(stat[i, j]), t))
            running = np.maximum(running, stat)
            times[(times == 0) & (stat >= brackets)] = t
            if t % 1000 == 0:
                print(
                    f"Training attempt{attempt} t={t}, remaining={(times==0).sum(axis=0)}, elapsed={time.monotonic()-start:.1f}s",
                    flush=True,
                )
            if np.all(times):
                break
        s.close()
        if not np.all(times):
            raise RuntimeError(
                "Training upper bracket failed to cross; no censored ARL is reported."
            )
        attempt_info = dict(
            attempt=attempt,
            upper_brackets=brackets.tolist(),
            upper_means=times.mean(axis=0).tolist(),
            longest=times.max(axis=0).tolist(),
            elapsed_seconds=time.monotonic() - start,
        )
        all_attempts.append(attempt_info)
        (OUT / f"global_external_training_attempt{attempt}.json").write_text(
            json.dumps(dict(info=attempt_info, records=records))
        )
        print("Training upper bracket:", attempt_info, flush=True)
        failed = times.mean(axis=0) < TARGET
        if failed.any():
            brackets += failed * np.array([0.6, 0.4])
            continue
        thresholds = []
        rows = []
        chosen_times = []
        for j, method in enumerate(METHODS):
            arrays = [np.asarray(r) for r in records[j]]

            def crossing(b):
                return np.array(
                    [a[np.searchsorted(a[:, 0], b), 1] for a in arrays], dtype=int
                )

            lo = min(a[0, 0] for a in arrays) - 1
            hi = brackets[j]
            for _ in range(40):
                mid = (lo + hi) / 2
                if crossing(mid).mean() < TARGET:
                    lo = mid
                else:
                    hi = mid
            values = crossing(hi)
            thresholds.append(float(hi))
            chosen_times.append(values)
            rows.append(
                dict(
                    method=method,
                    threshold=float(hi),
                    training_n=n,
                    training_seed=seed,
                    training_arl=float(values.mean()),
                    training_arl_se=estimate(values)[1],
                    censored=0,
                    upper_bracket=float(brackets[j]),
                )
            )
        np.savez_compressed(
            OUT / "global_external_training.npz", times=np.column_stack(chosen_times)
        )
        result = dict(
            thresholds=thresholds,
            rows=rows,
            analytic_start=approx,
            attempts=all_attempts,
        )
        (OUT / "global_external_calibration.json").write_text(
            json.dumps(result, indent=2)
        )
        write_csv("global_external_calibration.csv", rows)
        return np.array(thresholds)
    raise RuntimeError("Training bracket unsuccessful; all attempts retained.")


def validate(thresholds, n=750, seed=74100):
    rng = np.random.default_rng(seed)
    s = Statistics(n)
    times = np.zeros((n, 2), dtype=np.int32)
    start = time.monotonic()
    for t in range(1, 100001):
        active = (times[:, 0] == 0).astype(np.uint8) + 2 * (times[:, 1] == 0).astype(
            np.uint8
        )
        stat = s.step(rng.standard_normal((n, K)), active, t)
        times[(times == 0) & (stat >= thresholds)] = t
        if t % 1000 == 0:
            print(
                f"Validation t={t}, remaining={(times==0).sum(axis=0)}, elapsed={time.monotonic()-start:.1f}s",
                flush=True,
            )
        if np.all(times):
            break
    s.close()
    if not np.all(times):
        raise RuntimeError("Uncensored validation incomplete; increase limit.")
    rows = []
    for j, method in enumerate(METHODS):
        mean, se = estimate(times[:, j])
        rows.append(
            dict(
                method=method,
                n=n,
                seed=seed,
                threshold=thresholds[j],
                arl=mean,
                arl_se=se,
                censored=0,
                longest_run=int(times[:, j].max()),
            )
        )
    write_csv("global_external_arl.csv", rows)
    np.savez_compressed(OUT / "global_external_validation.npz", times=times)
    print("Validation:", rows, flush=True)
    return rows


def alternatives(thresholds, n=2000):
    allrows = []
    for i, mu in enumerate((0.4, 0.75)):
        thresholds_old = np.array(
            json.loads((OUT / f"calibration_mu{mu:g}.json").read_text())["thresholds"]
        )
        for j, (scenario, m) in enumerate((("sparse", 5), ("dense", 25))):
            seed = 50101 + 100 * i + j
            rng = np.random.default_rng(seed)
            s = Statistics(n)
            times = np.full((n, 2), H + 1, dtype=np.int32)

            old_times = np.full((n, 4), H + 1, dtype=np.int32)
            logr = np.full((n, K), -np.inf)
            w = np.zeros((n, K))
            logranks = np.log(np.arange(1, K + 1))
            for t in range(1, H + 1):
                x = rng.standard_normal((n, K))
                if t >= CHANGE:
                    x[:, :m] += mu
                active = (times[:, 0] > H).astype(np.uint8) + 2 * (
                    times[:, 1] > H
                ).astype(np.uint8)
                stat = s.step(x, active, t)
                times[(times > H) & (stat >= thresholds)] = t
                llr = mu * x - mu * mu / 2
                logr = np.logaddexp(0, logr) + llr
                w = np.maximum(0, w + llr)
                ordered = np.sort(logr, axis=1)[:, ::-1]
                old_stats = np.column_stack(
                    (
                        (ordered + logranks).max(axis=1),
                        ordered[:, 0],
                        w.max(axis=1),
                        w.sum(axis=1),
                    )
                )
                old_times[(old_times > H) & (old_stats >= thresholds_old)] = t
            s.close()
            saved = np.load(OUT / f"replicates_mu{mu:g}_rho0_{scenario}.npz")[
                "first_alarm"
            ][:, [2, 3, 4, 6]]
            np.testing.assert_array_equal(old_times, saved)
            combined = np.column_stack((old_times, times))
            names = (
                "e-d-BH (ours)",
                "e-d-Bonf./SR (ours)",
                "e-d-Bonf./CUSUM (ours)",
                "Sum-CUSUM",
            ) + METHODS
            np.savez_compressed(
                OUT / f"global_external_mu{mu:g}_{scenario}.npz",
                first_alarm=combined,
                methods=np.array(names),
                seed=seed,
                mu=mu,
                m=m,
            )
            for col, method in enumerate(names):
                v = combined[:, col]
                survive = v >= CHANGE
                delay = v[survive] - CHANGE + 1
                row = dict(
                    method=method,
                    mu=mu,
                    m=m,
                    scenario=scenario,
                    n=n,
                    seed=seed,
                    paired_old_first_alarms_verified=True,
                    conditional_n=int(survive.sum()),
                )
                for key, values in [
                    ("prechange_alarm", v < CHANGE),
                    ("conditional_restricted_delay", delay),
                    ("conditional_nondetection", v[survive] > H),
                ]:
                    row[key], row[key + "_se"] = estimate(values)
                allrows.append(row)
            print(
                f"Alternative mu={mu},m={m} complete; old observations/alarms reproduced exactly.",
                flush=True,
            )
            write_csv("global_external_summary.csv", allrows)
    return allrows


def summarize():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader((OUT / "global_external_summary.csv").open()))
    renamed = {
        "e-d-Bonferroni (ours)": "e-d-Bonf./SR (ours)",
        "Parallel CUSUM": "e-d-Bonf./CUSUM (ours)",
    }
    for row in rows:
        row["method"] = renamed.get(row["method"], row["method"])
    write_csv("global_external_summary.csv", rows)
    newarl = {
        r["method"]: r for r in csv.DictReader((OUT / "global_external_arl.csv").open())
    }
    oldarl = list(csv.DictReader((OUT / "arl_validation.csv").open()))
    oldnames = {
        "e-d-BH (ours)": "BH",
        "e-d-Bonf./SR (ours)": "SR-max",
        "e-d-Bonf./CUSUM (ours)": "CUSUM-max",
        "Sum-CUSUM": "Sum-CUSUM",
    }
    labels = list(oldnames) + list(METHODS)

    def arl(method, mu):
        if method in newarl:
            return newarl[method]
        return next(
            r
            for r in oldarl
            if r["method"] == oldnames[method]
            and r["calibration"] == "matched"
            and float(r["mu"]) == mu
        )

    arlrows = []
    for mu in (0.4, 0.75):
        for method in labels:
            arlrows.append(
                dict(
                    mu=mu,
                    method=method,
                    **{k: arl(method, mu)[k] for k in ("n", "arl", "arl_se")},
                )
            )
    write_csv("global_external_all_arl.csv", arlrows)
    lines = [
        "External global-detection comparison; all four planned alternatives retained.",
        "e-d-BH and e-d-Bonferroni with either SR or CUSUM inputs are our procedures.",
        "Actual validation ARLs are reported; fitting to target1000 does not guarantee exact matching.",
        "Xie--Siegmund estimates positive shifts; Chan fixes design shift1; existing methods use the true shift.",
        "Delay conditions on no alarm before101; nondetections by400 receive delay301.",
        "All alternative observations are exactly paired with the saved original Gaussian runs.",
        "",
        "method | mu | m | validated ARL (SE) | delay (SE) | P(prealarm) | P(nondetection | survival)",
    ]
    for row in rows:
        a = arl(row["method"], float(row["mu"]))
        lines.append(
            f"{row['method']} | {row['mu']} | {row['m']} | {float(a['arl']):.1f} ({float(a['arl_se']):.1f}) | {float(row['conditional_restricted_delay']):.2f} ({float(row['conditional_restricted_delay_se']):.2f}) | {float(row['prechange_alarm']):.4f} | {float(row['conditional_nondetection']):.4f}"
        )
    (OUT / "global_external_readout.txt").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    xpos = np.arange(len(labels))
    for ax, mu in zip(axes, (0.4, 0.75)):
        for m, offset, marker in [(5, -0.12, "o"), (25, 0.12, "s")]:
            selected = [
                next(
                    r
                    for r in rows
                    if r["method"] == method
                    and float(r["mu"]) == mu
                    and int(r["m"]) == m
                )
                for method in labels
            ]
            values = [float(r["conditional_restricted_delay"]) for r in selected]
            errors = [
                1.96 * float(r["conditional_restricted_delay_se"]) for r in selected
            ]
            ax.errorbar(
                xpos + offset,
                values,
                yerr=errors,
                fmt=marker,
                ms=5,
                capsize=2,
                label=f"{m} changed streams",
                color="#176B91" if m == 5 else "#C85454",
            )
        ax.set_xticks(
            xpos,
            [
                "e-d-BH\n(ours)",
                "e-d-Bonf./SR\n(ours)",
                "e-d-Bonf./CUSUM\n(ours)",
                "Sum-\nCUSUM",
                "Xie–Siegmund\nmixture GLR",
                "Chan\nCUSUM score",
            ],
            rotation=32,
            ha="right",
            fontsize=8,
        )
        ax.set_title(f"Gaussian shift {mu:g}")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Conditional detection delay ↓")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "global_external_delays.pdf", bbox_inches="tight")
    fig.savefig(OUT / "global_external_delays.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    write_table()
    print("\n".join(lines), flush=True)


def write_table():
    """Render only the table from saved results and calibrated thresholds."""
    rows = list(csv.DictReader((OUT / "global_external_summary.csv").open()))
    renamed = {
        "e-d-Bonferroni (ours)": "e-d-Bonf./SR (ours)",
        "Parallel CUSUM": "e-d-Bonf./CUSUM (ours)",
    }
    for row in rows:
        row["method"] = renamed.get(row["method"], row["method"])
    newarl = {
        r["method"]: r for r in csv.DictReader((OUT / "global_external_arl.csv").open())
    }
    oldarl = list(csv.DictReader((OUT / "arl_validation.csv").open()))
    oldnames = {
        "e-d-BH (ours)": "BH",
        "e-d-Bonf./SR (ours)": "SR-max",
        "e-d-Bonf./CUSUM (ours)": "CUSUM-max",
        "Sum-CUSUM": "Sum-CUSUM",
    }
    labels = list(oldnames) + list(METHODS)
    calibration = list(csv.DictReader((OUT / "calibration.csv").open()))
    external_calibration = {
        r["method"]: r
        for r in csv.DictReader((OUT / "global_external_calibration.csv").open())
    }

    def arl(method, mu):
        if method in newarl:
            return newarl[method]
        return next(
            r
            for r in oldarl
            if r["method"] == oldnames[method]
            and r["calibration"] == "matched"
            and float(r["mu"]) == mu
        )

    def threshold(method, mu):
        if method in external_calibration:
            h = float(external_calibration[method]["threshold"])
            assert h == float(newarl[method]["threshold"])
            return h
        row = next(
            r
            for r in calibration
            if r["method"] == oldnames[method] and float(r["mu"]) == mu
        )
        h = float(row["threshold"])

        saved = json.loads((OUT / f"calibration_mu{mu:g}.json").read_text())[
            "thresholds"
        ]
        assert h == saved[list(oldnames).index(method)]
        return h

    tex = [
        r"\begin{table}[tb]",
        r"\centering",
        r"\caption{Global detection with thresholds fitted to null ARL 1000. The fitted statistic threshold $h$ is shown to four decimal places; the statistic definitions are given in the text. For our procedures, the corresponding evidence thresholds are $e^h/r$ for BH (at rank $r$) and $e^h$ for Bonferroni. Other entries give independently validated ARL or conditional detection delay, with Monte Carlo standard errors in parentheses. Delay conditions on no alarm before time 101. Xie--Siegmund and Chan use 750 validation runs; the other methods use 1500. Every alternative uses the same 2000 runs.}",
        r"\label{tab:global-external}",
        r"\begingroup\fontsize{9}{11}\selectfont",
        r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Method & $h$ & Validated ARL & \multicolumn{2}{c}{Conditional delay $\downarrow$} \\",
        r" & & & 5 changes & 25 changes \\",
        r"\midrule",
    ]
    for mu in (0.4, 0.75):
        tex.append(r"\multicolumn{5}{l}{Shift $\mu=" + f"{mu:g}" + r"$} \\")
        tex.append(r"\addlinespace[2pt]")
        for method in labels:
            a = arl(method, mu)
            cells = []
            for m in (5, 25):
                r = next(
                    r
                    for r in rows
                    if r["method"] == method
                    and float(r["mu"]) == mu
                    and int(r["m"]) == m
                )
                cells.append(
                    f"{float(r['conditional_restricted_delay']):.2f} ({float(r['conditional_restricted_delay_se']):.2f})"
                )
            tex.append(
                f"{method} & {threshold(method,mu):.4f} & {float(a['arl']):.0f} ({float(a['arl_se']):.0f}) & "
                + " & ".join(cells)
                + r" \\"
            )
        if mu == 0.4:
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}", r"\endgroup", r"\end{table}"]
    (OUT / "global_external_table.tex").write_text("\n".join(tex) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--reuse-calibration", action="store_true")
    args = parser.parse_args()
    if args.tables_only:
        write_table()
        return
    if args.summarize_only:
        summarize()
        return
    start = time.time()
    check = checks_and_timing()
    if args.checks_only:
        return
    design = (ROOT / "global_external_design.txt").read_bytes()
    (OUT / "global_external_design_at_run.txt").write_bytes(design)
    meta = dict(
        design_sha256=hashlib.sha256(design).hexdigest(),
        python=platform.python_version(),
        numpy=np.__version__,
        K=K,
        window=W,
        target_arl=TARGET,
        checks=check,
        training_n=400,
        validation_n=750,
        alternative_n=2000,
        training_seed=73100,
        validation_seed=74100,
    )
    (OUT / "global_external_metadata.json").write_text(json.dumps(meta, indent=2))
    if args.reuse_calibration:
        thresholds = np.array(
            json.loads((OUT / "global_external_calibration.json").read_text())[
                "thresholds"
            ]
        )
    else:
        thresholds = calibrate()
    validate(thresholds)
    alternatives(thresholds)
    summarize()
    meta["elapsed_seconds"] = time.time() - start
    (OUT / "global_external_metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
