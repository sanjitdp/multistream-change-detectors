#!/usr/bin/env python3
"""Non-Gaussian simulations from diverse-design.txt."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
N = 1500
H = 400
Q = 0.05
ALPHA = Q / H
METHODS = ("BH", "SR", "CUSUM", "SHEREF")
M = len(METHODS)
LABELS = (
    "e-d-BH (ours)",
    "e-d-Bonferroni/SR (ours)",
    "e-d-Bonferroni/CUSUM (ours)",
    "SHEREF (fixed-time FDR)",
)
COLORS = ("#0072B2", "#D55E00", "#4D4D4D", "#009E73")
LINESTYLES = ("-", "--", "-.", ":")
SNAPSHOTS = (150, 200, 250, 300, 400)


def estimate(values):
    values = np.asarray(values, float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def add_est(row, key, values):
    row[key], row[key + "_se"] = estimate(values)


def add_ratio(row, key, error, patience):
    ratio = error.mean() / patience.mean()
    se = (error - ratio * patience).std(ddof=1) / (
        np.sqrt(len(error)) * patience.mean()
    )
    row[key], row[key + "_se"] = float(ratio), float(se)


def write_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(name):
    rows = list(csv.DictReader((OUT / name).open()))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def get(rows, **conditions):
    found = [
        r for r in rows if all(r[key] == value for key, value in conditions.items())
    ]
    assert len(found) == 1, (conditions, len(found))
    return found[0]


def bh_selection(logr, logb):
    ranks = np.arange(1, logr.shape[1] + 1)
    ordered = np.sort(logr, axis=1)[:, ::-1]
    r = np.where(ordered + np.log(ranks) >= logb, ranks, 0).max(axis=1)
    return logr >= (logb - np.log(np.maximum(r, 1)))[:, None]


def checks():
    assert np.array_equal(
        bh_selection(np.log([[6.0, 5.1, 0.1], [9.0, 0.1, 0.1]]), np.log(10.0)),
        [[True, True, False], [False, False, False]],
    )
    assert (0.5 * (1 + 0.5) + 0.5 * (1 - 0.5)) == 1.0
    for lam in (0.1, 1.0, 10.0):
        assert np.isclose(np.exp(lam * (1.75 - 1) - 0.75 * lam), 1.0)
    lr = np.array([0.4, 2.0, 1.5, 0.7])
    r = c = 0.0
    for t in range(len(lr)):
        r, c = (1 + r) * lr[t], max(1.0, c) * lr[t]
        windows = [np.prod(lr[j : t + 1]) for j in range(t + 1)]
        assert np.isclose(r, sum(windows)) and np.isclose(c, max(windows))


def run_case(model, k, scenario, rho, fraction, seed):
    rng = np.random.default_rng(seed)
    xi = np.full(k, H + 1, dtype=int)
    shifts = np.zeros(k)
    if model == "heavy":
        if scenario != "null":
            xi[:20] = 101
            shifts[:20] = np.tile([0.45, 0.9], 10)
            if scenario == "staggered":
                xi[10:20] = 201
        previous = None
    else:
        if scenario != "null":
            xi[: int(round(k * fraction))] = 101
        intercept = np.tile([0.1, 1.0], k // 2)
        previous = np.zeros((N, k), dtype=int)
        for _ in range(200):
            previous = rng.poisson(intercept + 0.3 * previous)
    affected = xi <= H
    m = int(affected.sum())
    logb = np.log(k / ALPHA)
    logr = np.full((N, k), -np.inf)
    logc = np.full((N, k), -np.inf)
    first = np.full((N, M), H + 1, dtype=np.int16)
    first5 = np.full((N, M), H + 1, dtype=np.int16)
    first80 = np.full((N, M), H + 1, dtype=np.int16)
    firstall = np.full((N, M), H + 1, dtype=np.int16)
    stream_times = np.full((N, M, k), H + 1, dtype=np.int16)
    stop_fdp, stop_v = np.zeros((N, M)), np.zeros((N, M))
    stop5_fdp, stop5_v = np.zeros((N, M)), np.zeros((N, M))
    base = dict(
        model=model,
        k=k,
        scenario=scenario,
        rho=rho,
        fraction=fraction,
        m=m,
        n=N,
        seed=seed,
    )
    curves = []
    snapshots = {}
    for t in range(1, H + 1):
        changed = t >= xi
        if model == "heavy":
            x = np.sqrt(1 - rho) * rng.standard_t(3, size=(N, k)) / np.sqrt(3)
            if rho:
                x += np.sqrt(rho) * rng.standard_t(3, size=(N, 1)) / np.sqrt(3)
            x += shifts * changed
            llr = np.log1p(0.5 * np.sign(x))
        else:
            lam = intercept + 0.3 * previous
            previous = rng.poisson(lam * np.where(changed, 1.75, 1.0))
            llr = previous * np.log(1.75) - 0.75 * lam
        logr = np.logaddexp(0, logr) + llr

        logc = np.maximum(0, logc) + llr
        selections = (
            bh_selection(logr, logb),
            logr >= logb,
            logc >= logb,
            bh_selection(logr, np.log(k * t / Q)),
        )
        for j, selected in enumerate(selections):
            r = selected.sum(axis=1)
            v = selected[:, ~changed].sum(axis=1)
            u = r - v
            fdp = v / np.maximum(r, 1)
            for num, times, fdp_at, v_at in (
                (1, first, stop_fdp, stop_v),
                (5, first5, stop5_fdp, stop5_v),
            ):
                fresh = (times[:, j] > H) & ((r >= num) | (t == H))
                times[fresh, j] = t
                fdp_at[fresh, j] = fdp[fresh]
                v_at[fresh, j] = v[fresh]
            if m:
                first80[(first80[:, j] > H) & (u >= math.ceil(0.8 * m)), j] = t
                firstall[(firstall[:, j] > H) & (u == m), j] = t
                fresh = selected & changed & (stream_times[:, j, :] > H)
                stream_times[:, j, :][fresh] = t
            if t % 5 == 0:
                row = dict(base, method=METHODS[j], t=t)
                add_est(row, "fdr", fdp)
                add_est(row, "pfer", v)
                add_est(row, "fwer", v > 0)
                if changed.any():
                    add_est(row, "tdr", u / changed.sum())
                    add_est(row, "prob80", u >= math.ceil(0.8 * changed.sum()))
                    add_est(row, "proball", u == changed.sum())
                curves.append(row)
            if t in SNAPSHOTS:
                snapshots[t, j] = (fdp, v, u)
    summary = []
    for j, method in enumerate(METHODS):
        row = dict(base, method=method)
        for t in SNAPSHOTS:
            fdp, v, u = snapshots[t, j]
            add_est(row, f"fdr_{t}", fdp)
            add_est(row, f"pfer_{t}", v)
            add_est(row, f"fwer_{t}", v > 0)
            mt = (xi <= t).sum()
            if mt:
                add_est(row, f"tdr_{t}", u / mt)
                add_est(row, f"prob80_{t}", u >= math.ceil(0.8 * mt))
                add_est(row, f"proball_{t}", u == mt)
        for label, times, err, count in (
            ("first", first, stop_fdp, stop_v),
            ("five", first5, stop5_fdp, stop5_v),
        ):
            add_est(row, f"bounded_{label}_mean", times[:, j])
            add_ratio(row, f"bounded_{label}_fdr_ratio", err[:, j], times[:, j])
            add_ratio(row, f"bounded_{label}_pfer_ratio", count[:, j], times[:, j])
        if m:
            add_est(
                row,
                "stream_restricted_delay",
                (stream_times[:, j, affected] - xi[affected] + 1).mean(axis=1),
            )
            add_est(
                row,
                "stream_nondetection",
                (stream_times[:, j, affected] > H).mean(axis=1),
            )
            for label, value in (("80", first80), ("all", firstall)):
                add_est(row, label + "_restricted_delay", value[:, j] - 101 + 1)
                add_est(row, label + "_nondetection", value[:, j] > H)
        summary.append(row)
    tag = f"diverse_{model}_k{k}_rho{rho:g}_{scenario}"
    np.savez_compressed(
        OUT / f"{tag}.npz",
        first=first,
        first5=first5,
        first80=first80,
        firstall=firstall,
        stream_times=stream_times,
        xi=xi,
        stop_fdp=stop_fdp,
        stop_v=stop_v,
        stop5_fdp=stop5_fdp,
        stop5_v=stop5_v,
        snapshot_fdp=np.stack(
            [snapshots[t, j][0] for t in SNAPSHOTS for j in range(M)]
        ),
        snapshot_v=np.stack([snapshots[t, j][1] for t in SNAPSHOTS for j in range(M)]),
        snapshot_u=np.stack([snapshots[t, j][2] for t in SNAPSHOTS for j in range(M)]),
    )
    return summary, curves


def setup_plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    return plt


def save_plot(plt, fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plots(summary, curves):
    plt = setup_plotting()
    for scenario in ("simultaneous", "staggered"):
        fig, axes = plt.subplots(2, 2, figsize=(6.35, 4.45), sharex=True, sharey="col")
        for row, rho in enumerate((0.0, 0.5)):
            for col, metric in enumerate(("fdr", "tdr")):
                ax = axes[row, col]
                for j, method in enumerate(METHODS):
                    records = [
                        r
                        for r in curves
                        if r["model"] == "heavy"
                        and r["rho"] == rho
                        and r["scenario"] == scenario
                        and r["method"] == method
                        and isinstance(r.get(metric), float)
                    ]
                    x = np.array([r["t"] for r in records])
                    y = np.array([r[metric] for r in records])
                    e = 1.96 * np.array([r[metric + "_se"] for r in records])
                    ax.plot(
                        x, y, color=COLORS[j], ls=LINESTYLES[j], lw=1.5, label=LABELS[j]
                    )
                    ax.fill_between(
                        x,
                        np.maximum(0, y - e),
                        np.minimum(1, y + e),
                        color=COLORS[j],
                        alpha=0.11,
                        linewidth=0,
                    )
                for change in [101] if scenario == "simultaneous" else [101, 201]:
                    ax.axvline(change, color=".6", ls=":", lw=0.8)
                ax.set_xlim(95, 400)
                ax.grid(axis="y", alpha=0.2, lw=0.5)
                if row == 0:
                    ax.set_title(
                        (
                            r"False discovery rate $\downarrow$"
                            if col == 0
                            else r"True discovery rate $\uparrow$"
                        ),
                        pad=8,
                    )
                if col == 0:
                    ax.set_ylabel(
                        ("Independent" if rho == 0 else "Correlated")
                        + rf" ($\rho={rho:g}$)"
                        + "\n"
                        + r"FDR at time $t$"
                    )
                    ax.set_ylim(bottom=-0.035 * ax.get_ylim()[1])
                else:
                    ax.set_ylabel(r"TDR at time $t$")
                    ax.set_ylim(0, 1.02)
                if row == 1:
                    ax.set_xlabel("Time $t$")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.015),
            ncol=2,
            frameon=False,
            fontsize=8.5,
            handlelength=2,
            columnspacing=1.7,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.925], h_pad=1.3, w_pad=1.7)
        save_plot(
            plt,
            fig,
            "diverse_heavy" if scenario == "simultaneous" else "diverse_staggered",
        )
    fig, axes = plt.subplots(2, 2, figsize=(6.35, 4.45), sharex=True, sharey="col")
    for row, scenario in enumerate(("sparse", "dense")):
        for col, metric in enumerate(("fdr_200", "tdr_200")):
            ax = axes[row, col]
            for j, method in enumerate(METHODS):
                records = [
                    get(summary, model="count", k=k, scenario=scenario, method=method)
                    for k in (20.0, 100.0, 500.0)
                ]
                x = np.array([20, 100, 500])
                y = np.array([r[metric] for r in records])
                e = 1.96 * np.array([r[metric + "_se"] for r in records])
                ax.errorbar(
                    x,
                    y,
                    yerr=np.stack((np.minimum(y, e), np.minimum(1 - y, e))),
                    color=COLORS[j],
                    ls=LINESTYLES[j],
                    marker="os^D"[j],
                    markersize=4,
                    lw=1.3,
                    capsize=2,
                    label=LABELS[j],
                )
            ax.set_xscale("log")
            ax.set_xticks([20, 100, 500], ["20", "100", "500"])
            ax.minorticks_off()
            ax.set_xlim(15, 650)
            ax.grid(axis="y", alpha=0.2, lw=0.5)
            if row == 0:
                ax.set_title(
                    (
                        r"False discovery rate $\downarrow$"
                        if col == 0
                        else r"True discovery rate $\uparrow$"
                    ),
                    pad=8,
                )
            if col == 0:
                ax.set_ylabel(
                    ("10%" if scenario == "sparse" else "50%")
                    + " changed\n"
                    + r"FDR at time 200"
                )
                ax.set_ylim(bottom=-0.035 * ax.get_ylim()[1])
            else:
                ax.set_ylabel(r"TDR at time 200")
                ax.set_ylim(0, 1.02)
            if row == 1:
                ax.set_xlabel("Number of streams $K$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handlelength=2,
        columnspacing=1.7,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.925], h_pad=1.3, w_pad=1.7)
    save_plot(plt, fig, "diverse_counts")


def main_plot(summary, curves):
    """Plot selected heavy-tail and count configurations."""
    plt = setup_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
    for j, method in enumerate(METHODS[:3]):
        records = [
            r
            for r in curves
            if r["model"] == "heavy"
            and r["rho"] == 0.5
            and r["scenario"] == "simultaneous"
            and r["method"] == method
            and isinstance(r.get("tdr"), float)
        ]
        x = np.array([r["t"] - 100 for r in records])
        y = np.array([r["tdr"] for r in records])
        e = 1.96 * np.array([r["tdr_se"] for r in records])
        axes[0].plot(x, y, color=COLORS[j], ls=LINESTYLES[j], lw=1.6, label=LABELS[j])
        axes[0].fill_between(
            x,
            np.maximum(0, y - e),
            np.minimum(1, y + e),
            color=COLORS[j],
            alpha=0.12,
            linewidth=0,
        )
        records = [
            get(summary, model="count", k=k, scenario="dense", method=method)
            for k in (20, 100, 500)
        ]
        y = np.array([r["tdr_200"] for r in records])
        e = 1.96 * np.array([r["tdr_200_se"] for r in records])
        axes[1].errorbar(
            [20, 100, 500],
            y,
            yerr=np.stack((np.minimum(y, e), np.minimum(1 - y, e))),
            color=COLORS[j],
            ls=LINESTYLES[j],
            marker="os^"[j],
            markersize=4.2,
            lw=1.5,
            capsize=2.3,
        )
    axes[0].set_title(
        r"(a) Heavy tails, $\rho=0.5$" + "\n" + r"True discovery rate $\uparrow$",
        fontsize=10,
        pad=8,
    )
    axes[1].set_title(
        "(b) Poisson counts, 50% changed\n" + r"True discovery rate $\uparrow$",
        fontsize=10,
        pad=8,
    )
    axes[0].set_xlim(0, 300)
    axes[0].set_xticks([0, 100, 200, 300])
    axes[0].set_xlabel("Post-change observations", fontsize=9.5)
    axes[0].set_ylabel("True discovery rate", fontsize=10)
    axes[1].set_xlabel("Number of streams $K$", fontsize=9.5)
    axes[1].set_xscale("log")
    axes[1].set_xticks([20, 100, 500], ["20", "100", "500"])
    axes[1].minorticks_off()
    axes[1].set_xlim(15, 650)
    for ax in axes:
        ax.set_ylim(0, 1.025)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", alpha=0.2, lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=2,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90], w_pad=1.6)
    save_plot(plt, fig, "diverse_main")


def output_checks(summary, curves):
    assert len(summary) == 15 * M and len(curves) == 15 * M * 80
    for row in curves:
        assert 0 <= row["fdr"] <= row["fwer"] + 1e-12 <= row["pfer"] + 1e-12
        if isinstance(row.get("tdr"), float):
            assert 0 <= row["proball"] <= row["prob80"] <= 1 and 0 <= row["tdr"] <= 1


def null_table(summary):
    """Write the global-null error table."""
    table = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Global-null FDR $\downarrow$ at time 200. Under the global null, this is the probability of any current declaration. Entries are estimates (Monte Carlo standard errors) from 1500 runs. The conditional mean of each Poisson count depends on its previous count. The SR threshold for e-d-BH is $8000K/r$; the e-d-Bonferroni thresholds for SR and CUSUM are $8000K$. SHEREF uses SR threshold $4000K/r$ at time 200, where $r$ is the step-up rank. Zero estimates and standard errors mean that no false declarations were observed at this time.}",
        r"\label{tab:diverse-null}",
        r"\begingroup\fontsize{9}{11}\selectfont",
        r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\hline",
        r" & & \multicolumn{4}{c}{FDR $\downarrow$} \\",
        r"Setting & $K$ & \shortstack{e-d-BH\\(ours)} & \shortstack{e-d-Bonferroni\\SR (ours)} & \shortstack{e-d-Bonferroni\\CUSUM (ours)} & SHEREF \\",
        r"\hline",
    ]
    cases = [("heavy", 100, rho, rf"Heavy tails, $\rho={rho:g}$") for rho in (0.0, 0.5)]
    cases += [("count", k, 0.0, "Poisson counts") for k in (20, 100, 500)]
    for model, k, rho, label in cases:
        cells = []
        for method in METHODS:
            r = get(summary, model=model, k=k, rho=rho, scenario="null", method=method)
            cells.append(f"${r['fdr_200']:.4f}\\;({r['fdr_200_se']:.4f})$")
        table.append(f"{label} & {k} & " + " & ".join(cells) + r" \\")
    table += [r"\hline", r"\end{tabular}", r"\endgroup", r"\end{table}"]
    (OUT / "diverse_null_table.tex").write_text("\n".join(table) + "\n")


def report(summary):
    lines = [
        "Measured diverse-study results (1500 replications per case)",
        "All FDR and TDR values below concern the current declaration set.",
        "Intervals shown in figures are +/-1.96 Monte Carlo standard errors.",
        "Finite-horizon budget q=.05,H400: alpha=q/H=.000125 for BH/SR/CUSUM.",
        "BH controls FDR at stopping rules bounded by400; SR/CUSUM control PFER there.",
        "SHEREF uses fixed-time FDR q=.05. All plotted empirical metrics are ordinary FDR/TDR.",
        "No equal achieved-FDR or empirically matched-ARL claim is made.",
        "",
    ]
    for scenario in ("null", "simultaneous", "staggered"):
        t = 300 if scenario == "staggered" else 200
        for rho in (0.0, 0.5):
            lines.append(f"Heavy tails: {scenario}, rho={rho:g}, reporting t={t}:")
            for method in METHODS:
                row = get(
                    summary, model="heavy", scenario=scenario, rho=rho, method=method
                )
                metrics = (f"fdr_{t}", f"tdr_{t}", f"prob80_{t}", f"proball_{t}")
                values = [
                    f"{key}={row[key]:.5f} (SE {row[key+'_se']:.5f})"
                    for key in metrics
                    if isinstance(row.get(key), (float, int))
                ]
                lines.append(
                    "  " + LABELS[METHODS.index(method)] + ": " + "; ".join(values)
                )
    for scenario in ("null", "sparse", "dense"):
        for k in (20.0, 100.0, 500.0):
            lines.append(
                f"Dependent Poisson counts: {scenario}, K={k:g}, reporting t=200:"
            )
            for method in METHODS:
                row = get(summary, model="count", scenario=scenario, k=k, method=method)
                metrics = ("fdr_200", "tdr_200", "prob80_200", "proball_200")
                values = [
                    f"{key}={row[key]:.5f} (SE {row[key+'_se']:.5f})"
                    for key in metrics
                    if isinstance(row.get(key), (float, int))
                ]
                lines.append(
                    "  " + LABELS[METHODS.index(method)] + ": " + "; ".join(values)
                )
    (OUT / "diverse_report.txt").write_text("\n".join(lines) + "\n")
    null_table(summary)

    bh = get(summary, model="heavy", scenario="simultaneous", rho=0.0, method="BH")
    sr = get(summary, model="heavy", scenario="simultaneous", rho=0.0, method="SR")
    sheref = get(
        summary, model="heavy", scenario="simultaneous", rho=0.0, method="SHEREF"
    )
    countbh = get(summary, model="count", k=500, scenario="dense", method="BH")
    countsr = get(summary, model="count", k=500, scenario="dense", method="SR")
    fragment = r"""% Generated from the prespecified experiments/diverse_benchmark.py study.
\subsection{Heavy-tailed observations and heterogeneous count streams}

Next, we consider two settings that use different input processes. Each configuration has 1500 independent runs, ending at time 400. We retain the thresholds $K/(r\alpha)$ for e-d-BH and $K/\alpha$ for the parallel SR and CUSUM charts, with $\alpha=q/H$, $q=0.05$ and $H=400$. This gives FDR control at level $q$ for e-d-BH at any stopping rule bounded by $H$; the parallel charts instead control PFER at level $q$. SHEREF uses $M_t^{(k)}/t$ at fixed-time FDR level $q$. We report the same ordinary FDR and TDR for every procedure, while keeping these different guarantees explicit.

For the first setting, let $Z_t^{(k)}$ and $U_t$ be independent Student-$t_3$ variables divided by $\sqrt{3}$, and set
\[
X_t^{(k)}=\mu_t^{(k)}+\sqrt{1-\rho}\,Z_t^{(k)}+\sqrt{\rho}\,U_t,
\qquad \rho\in\{0,0.5\}.
\]
There are 100 streams. Twenty change at time 101, with shifts alternating between 0.45 and 0.9; the others remain unchanged. We use the bounded factors $L_t^{(k)}=1+0.5\operatorname{sign}(X_t^{(k)})$. Conditional symmetry gives $\E[L_t^{(k)}\mid\mathcal F_{t-1}]=1$ for every still-null stream, including when the streams are correlated. Thus these inputs require neither a Student-$t$ likelihood nor a moment-generating function. We also run global-null configurations and staggered changes in ten streams at 101 and another ten at 201.

Figure~\ref{fig:diverse-heavy} shows the simultaneous-change results. Under independence, at time 200 e-d-BH has TDR @HEAVY_BH_TDR@ and FDR @HEAVY_BH_FDR@, compared with @HEAVY_SR_TDR@ and @HEAVY_SR_FDR@ for parallel SR. The probabilities of simultaneously identifying all 20 changed streams are @HEAVY_BH_ALL@ and @HEAVY_SR_ALL@. SHEREF has TDR @HEAVY_SHEREF_TDR@ and FDR @HEAVY_SHEREF_FDR@. Before $H$, its threshold $Kt/(rq)$ is lower than e-d-BH's $KH/(rq)$, so its higher fixed-time power is expected; its guarantee concerns deterministic times. The correlated, staggered and null configurations are also retained in the supplement.

\begin{figure}[tb]
\centering
\includegraphics[width=\textwidth]{experiments/results/diverse_heavy.pdf}
\caption{Ordinary FDR $\downarrow$ and TDR $\uparrow$ for heterogeneous shifts in 20 of 100 heavy-tailed streams. All changes occur at time 101. The rows use independent and correlated observation vectors. Shading gives 1.96 estimated Monte Carlo standard errors. The thresholds have different nominal error criteria, so both panels are needed to assess the tradeoff.}
\label{fig:diverse-heavy}
\end{figure}

For the second setting, observations are conditionally Poisson with intensity $c_t^{(k)}\lambda_t^{(k)}$, where $\lambda_t^{(k)}=a_k+0.3X_{t-1}^{(k)}$ and the known intercepts alternate between $a_k=0.1$ and $a_k=1$. The multiplier $c_t^{(k)}$ changes from 1 to 1.75. The streams are independent across $k$ and dependent over time. After 200 null warmup observations, monitoring begins with the conditional likelihood factors
\[
L_t^{(k)}=\exp\{X_t^{(k)}\log(1.75)-0.75\lambda_t^{(k)}\}.
\]
These factors have conditional mean one for still-null streams. We vary $K\in\{20,100,500\}$ and change either 10\% or 50\% of the streams at time 101, keeping equal numbers from the two intercept groups in each affected subset. All model parameters are known.

Figure~\ref{fig:diverse-counts} reports both error and identification at the prespecified time 200. With 50\% changed, e-d-BH's TDRs are @COUNT_BH_TDRS@ as $K$ increases, while parallel SR's are @COUNT_SR_TDRS@. At $K=500$, the respective FDRs are @COUNT_BH_FDR@ and @COUNT_SR_FDR@. The probability that e-d-BH identifies at least 80\% of the 250 changed streams is @COUNT_BH_80@; neither parallel chart reaches this fraction in the 1500 runs. No method identifies all 250 streams at this time. These comparisons illustrate how the step-up threshold uses evidence from other changed streams, while also showing the accompanying false-discovery cost. They do not establish dominance at a common FDR level.

\begin{figure}[tb]
\centering
\includegraphics[width=\textwidth]{experiments/results/diverse_counts.pdf}
\caption{Ordinary FDR $\downarrow$ and TDR $\uparrow$ at time 200 for heterogeneous, serially dependent count streams. Either 10\% or 50\% of the streams change at time 101. Bars show 1.96 estimated Monte Carlo standard errors; some lie inside the markers. The number of streams is on a logarithmic scale.}
\label{fig:diverse-counts}
\end{figure}

% Companion figure for the simulation-details appendix:
% experiments/results/diverse_staggered.pdf.
% Suggested caption: Ordinary FDR $\downarrow$ and TDR $\uparrow$ for staggered
% heterogeneous shifts in heavy-tailed streams. Ten streams change at101 and
% another ten at201. TDR is the fraction of already changed streams currently
% declared, so its denominator increases at the second change. Shading gives
% 1.96 estimated Monte Carlo standard errors. Other settings match the main
% heavy-tail experiment.
"""

    def rate(value):
        if value == 0:
            return "0 (no observed false discoveries)"
        if value < 0.001:
            coefficient, exponent = f"{value:.2e}".split("e")
            return rf"${coefficient}\times10^{{{int(exponent)}}}$"
        return f"{value:.4f}"

    values = {
        "HEAVY_BH_TDR": f"{bh['tdr_200']:.3f}",
        "HEAVY_BH_FDR": rate(bh["fdr_200"]),
        "HEAVY_SR_TDR": f"{sr['tdr_200']:.3f}",
        "HEAVY_SR_FDR": rate(sr["fdr_200"]),
        "HEAVY_BH_ALL": f"{bh['proball_200']:.3f}",
        "HEAVY_SR_ALL": f"{sr['proball_200']:.3f}",
        "HEAVY_SHEREF_TDR": f"{sheref['tdr_200']:.3f}",
        "HEAVY_SHEREF_FDR": rate(sheref["fdr_200"]),
        "COUNT_BH_FDR": rate(countbh["fdr_200"]),
        "COUNT_SR_FDR": rate(countsr["fdr_200"]),
        "COUNT_BH_80": f"{countbh['prob80_200']:.3f}",
        "COUNT_BH_TDRS": ", ".join(
            f"{get(summary, model='count', k=k, scenario='dense', method='BH')['tdr_200']:.3f}"
            for k in (20, 100, 500)
        ),
        "COUNT_SR_TDRS": ", ".join(
            f"{get(summary, model='count', k=k, scenario='dense', method='SR')['tdr_200']:.3f}"
            for k in (20, 100, 500)
        ),
    }
    for key, value in values.items():
        fragment = fragment.replace("@" + key + "@", value)
    assert "@" not in fragment
    (OUT / "diverse_results.tex").write_text(fragment)
    appendix = r"""% Exact model and protocol details for the non-Gaussian experiments.
\subsection{Further non-Gaussian simulations}
\label{app:diverse-details}

Each of the following 15 configurations uses 1500 independent runs and a monitoring horizon of $H=400$. We set $q=0.05$ and $\alpha=q/H=0.000125$. Given factors $L_t^{(k)}$, the SR and restart-CUSUM statistics are
\[
M_t^{(k)}=(1+M_{t-1}^{(k)})L_t^{(k)},\qquad
C_t^{(k)}=\max\{1,C_{t-1}^{(k)}\}L_t^{(k)},
\quad M_0^{(k)}=C_0^{(k)}=0.
\]
The latter is the maximum product over nonempty restart windows and can be smaller than one. e-d-BH uses the SR inputs and thresholds $K/(r\alpha)$. Parallel SR (e-d-Bonferroni) and parallel CUSUM use threshold $K/\alpha$. The resulting guarantees concern FDR for e-d-BH and PFER for the parallel charts, each at level $q$ for any stopping time bounded by $H$. We also run SHEREF with inputs $M_t^{(k)}/t$ and fixed-time FDR level $q$. Its threshold $Kt/(rq)$ is smaller than e-d-BH's $KH/(rq)$ before $H$, and the declaration sets agree at $H$. Its earlier identification therefore does not establish the same guarantee at a data-dependent stopping time.

For the heavy-tailed experiment, let all $Z_t^{(k)}$ and $U_t$ be independent Student-$t_3$ variables divided by $\sqrt{3}$, and define
\[
X_t^{(k)}=\mu_t^{(k)}+\sqrt{1-\rho}\,Z_t^{(k)}+\sqrt{\rho}\,U_t,
\qquad K=100,\quad\rho\in\{0,0.5\}.
\]
Vectors are independent over time. The six configurations are the global null, simultaneous changes in streams 1--20 at time 101, and staggered changes in streams 1--10 at time 101 and streams 11--20 at time 201, each at both values of $\rho$. Changed streams have shifts alternating between 0.45 and 0.9, so each group of ten contains five of each size. The remaining streams have mean zero throughout. The factors are
\[
L_t^{(k)}=1+0.5\operatorname{sign}(X_t^{(k)}).
\]
For every still-null stream, conditional symmetry gives $\E[L_t^{(k)}\mid\mathcal F_{t-1}]=1$. This construction uses symmetry, rather than the Student-$t$ likelihood or an exponential-moment assumption. Figures~\ref{fig:diverse-heavy-full} and~\ref{fig:diverse-staggered-full} report both ordinary FDR and TDR.

For the count experiment, $K\in\{20,100,500\}$ and the known intercepts alternate between $a_k=0.1$ and $a_k=1$. Streams are independent across $k$. Conditional on the observed past, let
\[
\lambda_t^{(k)}=a_k+0.3X_{t-1}^{(k)},\qquad
X_t^{(k)}\sim\operatorname{Poisson}(c_t^{(k)}\lambda_t^{(k)}).
\]
The multiplier $c_t^{(k)}$ is one before a change and 1.75 afterwards. Since $0.3(1.75)<1$, both regimes have finite stationary means. Starting with $X=0$, we generate 200 null warmup observations and retain the last count as observed initial history; only the detector statistics are reset. At each $K$, we run the global null and configurations in which the first 10\% or 50\% of streams change at time 101. Every affected subset has equal numbers from both intercept groups. These give nine configurations. The exact conditional likelihood factors are
\[
L_t^{(k)}=\exp\{X_t^{(k)}\log(1.75)-0.75\lambda_t^{(k)}\}.
\]
They have conditional mean one for every still-null stream, despite the serial dependence. The intensity model and all its parameters are known. Figure~\ref{fig:diverse-counts-full} reports the prespecified time 200 for both affected fractions.

These comparisons separate two effects. The sign factors and conditional Poisson factors show that the input construction need not use an independent Gaussian model. The count experiment also varies the number of streams while keeping the affected fraction and composition fixed. If the number of declarations is proportional to $K$, the step-up threshold $K/(r\alpha)$ remains of constant order; the parallel-chart threshold $K/\alpha$ grows with $K$. This explains the different identification patterns, without implying dominance at a common achieved FDR. Controlling the total number of false discoveries is more restrictive here than controlling their proportion, and the plots retain that false-discovery cost.

All errors and powers concern the current declaration set. TDR is the mean fraction of already changed streams currently declared; its denominator increases at time 201 in the staggered experiment. Curves are saved every five observations. Reporting-time summaries at 150, 200, 250, 300 and 400, including probabilities of identifying 80\% or all changed streams, are provided with the code. Monte Carlo standard errors treat each run as an independent unit, retaining dependence among its streams. Shading and bars show 1.96 estimated standard errors, with rate intervals clipped to $[0,1]$. An observed error rate of zero does not establish that the true rate is zero. No configuration was removed.

\begin{figure}[tb]
\centering
\includegraphics[width=\textwidth]{experiments/results/diverse_heavy.pdf}
\caption{Ordinary FDR $\downarrow$ and TDR $\uparrow$ for simultaneous heterogeneous shifts in 20 of 100 heavy-tailed streams at time 101. Rows use independent and correlated vectors. The horizon is 400; the nominal bounded-horizon budget is $q=0.05$ for FDR (e-d-BH) or PFER (parallel charts). SHEREF instead uses fixed-time FDR level 0.05. Shading shows 1.96 Monte Carlo standard errors.}
\label{fig:diverse-heavy-full}
\end{figure}

\begin{figure}[tb]
\centering
\includegraphics[width=\textwidth]{experiments/results/diverse_staggered.pdf}
\caption{The same comparison with ten heterogeneous shifts at time 101 and another ten at 201. TDR $\uparrow$ counts only already changed streams, so its denominator increases at the second change. Ordinary FDR $\downarrow$, error budgets and uncertainty are defined as in Figure~\ref{fig:diverse-heavy-full}.}
\label{fig:diverse-staggered-full}
\end{figure}

\begin{figure}[tb]
\centering
\includegraphics[width=\textwidth]{experiments/results/diverse_counts.pdf}
\caption{Ordinary FDR $\downarrow$ and TDR $\uparrow$ at time 200 for heterogeneous, serially dependent count streams. Either 10\% or 50\% change at time 101. The number of streams is on a logarithmic scale. Error budgets are as in Figure~\ref{fig:diverse-heavy-full}. Bars show 1.96 Monte Carlo standard errors; some lie inside the markers.}
\label{fig:diverse-counts-full}
\end{figure}

\input{experiments/results/diverse_null_table.tex}
"""
    (OUT / "diverse_appendix.tex").write_text(appendix)
    main_caption = r"""\begin{figure}[tb]
\centering
\includegraphics[width=\textwidth]{experiments/results/diverse_main.pdf}
\caption{Identification with non-Gaussian inputs. Left: 20 heterogeneous shifts in 100 heavy-tailed streams with cross-stream correlation 0.5. Right: TDR $\uparrow$ at time 200 when 50\% of heterogeneous, serially dependent count streams change at time 101. All three rules use bounded-horizon budget $q=0.05$ over 400 observations, for FDR (e-d-BH) or PFER (parallel charts). Shading and bars give 1.96 Monte Carlo standard errors from 1500 runs. Ordinary FDR, SHEREF and the remaining configurations are reported in Appendix~\ref{app:diverse-details}.}
\label{fig:diverse-main}
\end{figure}
"""
    (OUT / "diverse_main_caption.tex").write_text(main_caption)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    checks()
    if args.summarize_only:
        summary, curves = read_csv("diverse_summary.csv"), read_csv(
            "diverse_curves.csv"
        )
    else:
        start = time.time()
        summary, curves = [], []
        metadata = dict(
            design_sha256=hashlib.sha256(
                (ROOT / "diverse-design.txt").read_bytes()
            ).hexdigest(),
            n=N,
            horizon=H,
            alpha=ALPHA,
            q=Q,
            methods=METHODS,
            python=platform.python_version(),
            numpy=np.__version__,
        )
        (OUT / "diverse_metadata.json").write_text(json.dumps(metadata, indent=2))
        for i, rho in enumerate((0.0, 0.5)):
            for j, scenario in enumerate(("null", "simultaneous", "staggered")):
                a, b = run_case(
                    "heavy",
                    100,
                    scenario,
                    rho,
                    0.0 if scenario == "null" else 0.2,
                    61000 + 10 * i + j,
                )
                summary.extend(a)
                curves.extend(b)
                write_csv("diverse_summary.csv", summary)
                write_csv("diverse_curves.csv", curves)
                print(
                    f"Finished heavy {scenario},rho={rho}; {time.time()-start:.0f}s",
                    flush=True,
                )
        for i, k in enumerate((20, 100, 500)):
            for j, (scenario, fraction) in enumerate(
                (("null", 0.0), ("sparse", 0.1), ("dense", 0.5))
            ):
                a, b = run_case("count", k, scenario, 0.0, fraction, 62000 + 10 * i + j)
                summary.extend(a)
                curves.extend(b)
                write_csv("diverse_summary.csv", summary)
                write_csv("diverse_curves.csv", curves)
                print(
                    f"Finished count {scenario},K={k}; {time.time()-start:.0f}s",
                    flush=True,
                )
        metadata["elapsed_seconds"] = time.time() - start
        (OUT / "diverse_metadata.json").write_text(json.dumps(metadata, indent=2))
    output_checks(summary, curves)
    plots(summary, curves)
    main_plot(summary, curves)
    report(summary)
    print(
        "Diverse study complete; all 15 prespecified configurations retained.",
        flush=True,
    )


if __name__ == "__main__":
    main()
