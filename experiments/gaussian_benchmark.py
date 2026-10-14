#!/usr/bin/env python3
"""Gaussian identification and average-run-length comparisons."""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
K = 50
ALPHA = 0.001
CHANGE = 101
HORIZON = 400
ARL_TARGET = 1000
METHODS = ("BH", "SR-max", "CUSUM-max", "Sum-CUSUM")
LOG_RANKS = np.log(np.arange(1, K + 1))


def selections(logr, log_threshold, bh):
    if not bh:
        return logr >= log_threshold
    ordered = np.sort(logr, axis=1)[:, ::-1]
    ranks = np.arange(1, logr.shape[1] + 1)
    r = np.where(ordered + np.log(ranks) >= log_threshold, ranks, 0).max(axis=1)
    return logr >= (log_threshold - np.log(np.maximum(r, 1)))[:, None]


def stats(logr, w):
    ordered = np.sort(logr, axis=1)[:, ::-1]
    return np.stack(
        (
            (ordered + LOG_RANKS).max(axis=1),
            ordered[:, 0],
            w.max(axis=1),
            w.sum(axis=1),
        ),
        axis=1,
    )


def estimate(x):
    x = np.asarray(x, dtype=float)
    return (float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size)))


def ratio_estimate(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ratio = x.mean() / y.mean()
    se = (x - ratio * y).std(ddof=1) / (np.sqrt(x.size) * y.mean())
    return float(ratio), float(se)


def add_est(row, name, x):
    row[name], row[name + "_se"] = estimate(x)


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def calibrate(mu, n, horizon, seed):
    """Fit thresholds using null-path ARLs capped at horizon + 1."""
    rng = np.random.default_rng(seed)
    logr = np.full((n, K), -np.inf)
    w = np.zeros((n, K))
    running = np.full((n, 4), -np.inf)
    history = np.empty((4, n, horizon), dtype=np.float32)
    for t in range(horizon):
        llr = mu * rng.standard_normal((n, K)) - mu * mu / 2
        logr = np.logaddexp(0, logr) + llr
        w = np.maximum(0, w + llr)
        running = np.maximum(running, stats(logr, w))
        history[:, :, t] = running.T
    thresholds = []
    training_rows = []
    for j, method in enumerate(METHODS):
        h = history[j]
        lo, hi = float(h.min()), float(h.max()) + 1
        for _ in range(25):
            middle = (lo + hi) / 2
            crossed = h >= middle
            times = np.where(crossed[:, -1], crossed.argmax(axis=1) + 1, horizon + 1)
            if times.mean() < ARL_TARGET:
                lo = middle
            else:
                hi = middle
        thresholds.append(hi)
        crossed = h >= hi
        times = np.where(crossed[:, -1], crossed.argmax(axis=1) + 1, horizon + 1)
        training_rows.append(
            dict(
                mu=mu,
                method=method,
                threshold=hi,
                implied_eop_level=K * np.exp(-hi) if j < 2 else "",
                training_restricted_arl=float(times.mean()),
                training_censored=int((times > horizon).sum()),
                training_n=n,
                training_horizon=horizon,
                seed=seed,
            )
        )
    return np.array(thresholds), training_rows


def validate_arl(mu, thresholds, n, seed, limit=100000):
    rng = np.random.default_rng(seed)
    ids = np.arange(n)
    logr = np.full((n, K), -np.inf)
    w = np.zeros((n, K))
    times = np.zeros((n, 6), dtype=int)
    for t in range(1, limit + 1):
        llr = mu * rng.standard_normal(logr.shape) - mu * mu / 2
        logr = np.logaddexp(0, logr) + llr
        w = np.maximum(0, w + llr)
        s = stats(logr, w)
        alarms = np.column_stack((s >= thresholds, s[:, :2] >= np.log(K / ALPHA)))
        old = times[ids]
        old[(old == 0) & alarms] = t
        times[ids] = old
        active = (old == 0).any(axis=1)
        ids, logr, w = ids[active], logr[active], w[active]
        if not ids.size:
            break
    if ids.size:
        raise RuntimeError(
            "ARL validation censored; increase limit instead of reporting ARL."
        )
    rows = []
    labels = list(METHODS) + ["BH", "SR-max"]
    for j, method in enumerate(labels):
        row = dict(
            mu=mu,
            calibration="matched" if j < 4 else "theory",
            method=method,
            n=n,
            seed=seed,
            censored=0,
            longest_run=int(times[:, j].max()),
        )
        add_est(row, "arl", times[:, j])
        rows.append(row)
    np.savez_compressed(OUT / f"arl_mu{mu:g}.npz", times=times)
    return rows


def run_scenario(mu, rho, scenario, thresholds, n, seed):
    xi = np.full(K, HORIZON + 1, dtype=int)
    if scenario == "sparse":
        xi[:5] = CHANGE
    elif scenario == "dense":
        xi[:25] = CHANGE
    elif scenario == "staggered":
        xi[:10] = CHANGE
        xi[10:25] = 151
    elif scenario != "null":
        raise ValueError(scenario)
    affected = xi <= HORIZON
    total_affected = int(affected.sum())
    rng = np.random.default_rng(seed)
    logr = np.full((n, K), -np.inf)
    w = np.zeros((n, K))

    names = ("BH", "SR-max", "BH", "SR-max", "CUSUM-max", "SR/t BH", "Sum-CUSUM")
    calibrations = (
        "theory",
        "theory",
        "matched",
        "matched",
        "matched",
        "fixed-time",
        "matched",
    )
    count = len(names)
    first_alarm = np.full((n, count), HORIZON + 1, dtype=int)
    first_true = np.full((n, count), HORIZON + 1, dtype=int)
    first_80 = first_true.copy()
    first_all = first_true.copy()

    stop_fdp = np.zeros((n, count))
    stop_v = np.zeros((n, count))
    first_five = np.full((n, count), HORIZON + 1, dtype=int)
    five_fdp = np.zeros((n, count))
    five_v = np.zeros((n, count))
    stream_times = np.full((n, 6, K), HORIZON + 1, dtype=int)
    curve_rows = []
    snapshots = {}
    for t in range(1, HORIZON + 1):
        noise = rng.standard_normal((n, K))
        if rho:
            noise = np.sqrt(1 - rho) * noise + np.sqrt(rho) * rng.standard_normal(
                (n, 1)
            )
        changed = t >= xi
        llr = mu * (noise + mu * changed) - mu * mu / 2
        logr = np.logaddexp(0, logr) + llr
        w = np.maximum(0, w + llr)
        masks = [
            selections(logr, np.log(K / ALPHA), True),
            selections(logr, np.log(K / ALPHA), False),
            selections(logr, thresholds[0], True),
            selections(logr, thresholds[1], False),
            w >= thresholds[2],
            selections(logr, np.log(K * t / 0.05), True),
        ]
        global_alarm = w.sum(axis=1) >= thresholds[3]
        for j in range(count):
            if j < 6:
                selected = masks[j]
                r = selected.sum(axis=1)
                v = selected[:, ~changed].sum(axis=1)
                u = r - v
                fdp = v / np.maximum(r, 1)
                alarm = r > 0
                true_alarm = u > 0
                new_stream = selected & changed & (stream_times[:, j, :] > HORIZON)
                stream_times[:, j, :][new_stream] = t
                if total_affected:
                    first_80[
                        (first_80[:, j] > HORIZON)
                        & (u >= math.ceil(0.8 * total_affected)),
                        j,
                    ] = t
                    first_all[
                        (first_all[:, j] > HORIZON) & (u == total_affected), j
                    ] = t
                new_five = (first_five[:, j] > HORIZON) & ((r >= 5) | (t == HORIZON))
                first_five[new_five, j] = t
                five_fdp[new_five, j] = fdp[new_five]
                five_v[new_five, j] = v[new_five]
            else:
                alarm = global_alarm

                true_alarm = alarm & bool(changed.any())
                v = alarm.astype(float) * (not changed.any())
                fdp = v
            new_alarm = (first_alarm[:, j] > HORIZON) & (alarm | (t == HORIZON))

            first_alarm[new_alarm & alarm, j] = t
            stop_fdp[new_alarm, j] = fdp[new_alarm]
            stop_v[new_alarm, j] = v[new_alarm]
            new_true = (first_true[:, j] > HORIZON) & true_alarm
            first_true[new_true, j] = t
            if j < 6 and (t % 5 == 0 or t == CHANGE):
                row = dict(
                    mu=mu,
                    rho=rho,
                    scenario=scenario,
                    method=names[j],
                    calibration=calibrations[j],
                    t=t,
                    n=n,
                )
                add_est(row, "fdr", fdp)
                add_est(row, "pfer", v)
                add_est(row, "fwer", v > 0)
                if changed.any():
                    add_est(row, "tdr", u / changed.sum())
                    add_est(row, "prob80", u >= math.ceil(0.8 * changed.sum()))
                    add_est(row, "proball", u == changed.sum())
                curve_rows.append(row)
            if j < 6 and t in (100, 120, 150, 200, 300, 400):
                snapshots[(t, j)] = (fdp.copy(), v.copy(), u.copy())
    rows = []
    for j in range(count):
        row = dict(
            mu=mu,
            rho=rho,
            scenario=scenario,
            method=names[j],
            calibration=calibrations[j],
            n=n,
            seed=seed,
        )
        bounded_tau = np.minimum(first_alarm[:, j], HORIZON)
        add_est(row, "bounded_first_mean", bounded_tau)
        add_est(row, "bounded_first_fdr", stop_fdp[:, j])
        add_est(row, "bounded_first_pfer", stop_v[:, j])
        row["bounded_first_fdr_ratio"], row["bounded_first_fdr_ratio_se"] = (
            ratio_estimate(stop_fdp[:, j], bounded_tau)
        )
        row["bounded_first_pfer_ratio"], row["bounded_first_pfer_ratio_se"] = (
            ratio_estimate(stop_v[:, j], bounded_tau)
        )
        add_est(row, "prechange_alarm", first_alarm[:, j] < CHANGE)
        if j < 6:
            row["bounded_five_fdr_ratio"], row["bounded_five_fdr_ratio_se"] = (
                ratio_estimate(five_fdp[:, j], first_five[:, j])
            )
            row["bounded_five_pfer_ratio"], row["bounded_five_pfer_ratio_se"] = (
                ratio_estimate(five_v[:, j], first_five[:, j])
            )
        if total_affected:

            add_est(row, "first_true_restricted_delay", first_true[:, j] - CHANGE + 1)
            add_est(row, "first_true_nondetection", first_true[:, j] > HORIZON)
            survived = first_alarm[:, j] >= CHANGE
            add_est(
                row,
                "global_conditional_restricted_delay",
                first_alarm[survived, j] - CHANGE + 1,
            )
            row["global_conditional_n"] = int(survived.sum())
            add_est(
                row,
                "global_conditional_nondetection",
                first_alarm[survived, j] > HORIZON,
            )
            if j < 6:
                for label, val in (("80", first_80), ("all", first_all)):
                    add_est(row, f"{label}_restricted_delay", val[:, j] - CHANGE + 1)
                    add_est(row, f"{label}_nondetection", val[:, j] > HORIZON)
                delays = stream_times[:, j, affected] - xi[affected] + 1
                add_est(row, "stream_restricted_delay", delays.mean(axis=1))
                add_est(
                    row,
                    "stream_nondetection",
                    (stream_times[:, j, affected] > HORIZON).mean(axis=1),
                )
        for t in (100, 120, 150, 200, 300, 400):
            if j < 6:
                fdp, v, u = snapshots[t, j]
                add_est(row, f"fdr_{t}", fdp)
                add_est(row, f"pfer_{t}", v)
                if (xi <= t).any():
                    add_est(row, f"tdr_{t}", u / (xi <= t).sum())
                    add_est(row, f"prob80_{t}", u >= math.ceil(0.8 * (xi <= t).sum()))
                    add_est(row, f"proball_{t}", u == (xi <= t).sum())
        rows.append(row)
    tag = f"mu{mu:g}_rho{rho:g}_{scenario}"
    np.savez_compressed(
        OUT / f"replicates_{tag}.npz",
        first_alarm=first_alarm,
        first_true=first_true,
        first_80=first_80,
        first_all=first_all,
        stop_fdp=stop_fdp,
        stop_v=stop_v,
        first_five=first_five,
        five_fdp=five_fdp,
        five_v=five_v,
        stream_times=stream_times,
        xi=xi,
    )
    return rows, curve_rows


def checks():
    a = np.log([[6.1, 5.1, 0.1], [9.0, 0.1, 0.1], [10.0, 0.1, 0.1]])
    assert np.array_equal(
        selections(a, np.log(10.0), True),
        [[True, True, False], [False, False, False], [True, False, False]],
    )

    lr = np.array([0.4, 2.0, 1.5, 0.7])
    r = 0.0
    for t in range(len(lr)):
        r = (r + 1) * lr[t]
        direct = sum(np.prod(lr[j : t + 1]) for j in range(t + 1))
        assert np.isclose(r, direct)

    w = 0.0
    for t in range(len(lr)):
        w = max(0.0, w + np.log(lr[t]))
        assert np.isclose(
            w, max([0.0] + [np.log(lr[j : t + 1]).sum() for j in range(t + 1)])
        )
    assert ratio_estimate([0, 1], [1, 3])[0] == 0.25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--n-calibration", type=int, default=800)
    parser.add_argument("--n-arl", type=int, default=1500)
    parser.add_argument("--calibration-horizon", type=int, default=10000)
    parser.add_argument("--reuse-calibration", action="store_true")
    args = parser.parse_args()
    checks()
    OUT.mkdir(exist_ok=True)
    start = time.time()
    calibration_rows, arl_rows, summary_rows, curve_rows = [], [], [], []
    metadata = dict(
        K=K,
        alpha=ALPHA,
        change=CHANGE,
        horizon=HORIZON,
        arl_target=ARL_TARGET,
        arguments=vars(args),
        python=platform.python_version(),
        numpy=np.__version__,
    )
    for i, mu in enumerate((0.4, 0.75)):
        calfile = OUT / f"calibration_mu{mu:g}.json"
        if args.reuse_calibration and calfile.exists():
            cal = json.loads(calfile.read_text())
            thresholds = np.array(cal["thresholds"])
            rows = cal["rows"]
        else:
            print(f"Calibrating mu={mu}", flush=True)
            thresholds, rows = calibrate(
                mu, args.n_calibration, args.calibration_horizon, 28100 + i
            )
            calfile.write_text(
                json.dumps(dict(thresholds=thresholds.tolist(), rows=rows), indent=2)
            )
        calibration_rows += rows
        print(f"Thresholds mu={mu}: {thresholds}", flush=True)
        arl_rows += validate_arl(mu, thresholds, args.n_arl, 39100 + i)
        write_csv("calibration.csv", calibration_rows)
        write_csv("arl_validation.csv", arl_rows)
        print(f"Validated mu={mu}; elapsed {time.time()-start:.0f}s", flush=True)
        for j, rho in enumerate((0.0, 0.5)):
            for k, scenario in enumerate(("null", "sparse", "dense", "staggered")):
                rows, curves = run_scenario(
                    mu, rho, scenario, thresholds, args.n, 50100 + 100 * i + 10 * j + k
                )
                summary_rows += rows
                curve_rows += curves
                write_csv("summary.csv", summary_rows)
                write_csv("curves.csv", curve_rows)
                print(
                    f"Finished mu={mu}, rho={rho}, {scenario}; elapsed {time.time()-start:.0f}s",
                    flush=True,
                )
    metadata["elapsed_seconds"] = time.time() - start
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("All simulations complete.", flush=True)


if __name__ == "__main__":
    main()
