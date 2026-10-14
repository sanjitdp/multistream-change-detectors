#!/usr/bin/env python3
"""Bounded-reporting simulations from tradeoff_design.txt."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib_tradeoff"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

K, H, NU, MU, Q = 100, 400, 101, 0.5, 0.05
METHODS = ("e-d-BH (ours)", "Parallel SR", "Per-stream SR", "SR/t e-BH (SHEREF)")
DISPLAY_LABELS = (
    "e-d-BH (ours)",
    "e-d-Bonferroni (ours)",
    "Per-stream SR",
    "SHEREF (fixed-time FDR)",
)
REPORTS = ("first_1_capped", "first_10_capped", "fixed_H")
COLORS = ("#2166ac", "#d97918", "#a84266", "#777777")
RANKS = np.arange(1, K + 1)


def estimate(values):
    x = np.asarray(values, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))


def add_est(row, name, x):
    row[name], row[name + "_se"] = estimate(x)


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def bh_selected(evidence, ordered_times_rank, family_threshold):
    r = np.where(ordered_times_rank >= family_threshold, RANKS, 0).max(axis=1)
    cutoff = family_threshold / np.maximum(r, 1)
    return (evidence >= cutoff[:, None]) & (r[:, None] > 0)


def simulate(m, n):
    seed = 61006200 + m
    rng = np.random.default_rng(seed)
    evidence = np.zeros((n, K), dtype=np.float64)

    report = np.zeros((n, len(METHODS), len(REPORTS), 4), dtype=np.int32)

    recovery = np.full((n, len(METHODS), 2), H + 1, dtype=np.int32)
    curves = []
    for t in range(1, H + 1):
        x = rng.standard_normal((n, K))
        if t >= NU and m:
            x[:, :m] += MU
        evidence = (1 + evidence) * np.exp(MU * x - MU * MU / 2)
        order_rank = np.sort(evidence, axis=1)[:, ::-1] * RANKS
        selected = (
            bh_selected(evidence, order_rank, K * H / Q),
            evidence >= K * H / Q,
            evidence >= H / Q,
            bh_selected(evidence, order_rank, K * t / Q),
        )

        if t == H:
            assert not np.any(selected[1] & ~selected[0])
        for j, declared in enumerate(selected):
            size = declared.sum(axis=1)
            true = (
                declared[:, :m].sum(axis=1) if t >= NU and m else np.zeros(n, dtype=int)
            )
            false = size - true
            for s, r in enumerate((1, 10, None)):
                newly = (report[:, j, s, 0] == 0) & (
                    (size >= r) if r is not None else False
                )
                if t == H:
                    newly = report[:, j, s, 0] == 0
                report[newly, j, s] = np.column_stack(
                    (np.full(newly.sum(), t), size[newly], false[newly], true[newly])
                )
            if t >= NU and m:
                for f, fraction in enumerate((0.8, 1.0)):
                    newly = (recovery[:, j, f] == H + 1) & (
                        true >= int(np.ceil(fraction * m))
                    )
                    recovery[newly, j, f] = t
            if t >= NU:
                row = dict(
                    m=m,
                    method=METHODS[j],
                    t=t,
                    post_observations=t - NU + 1,
                    n=n,
                    seed=seed,
                )
                add_est(row, "tdr", true / m if m else np.zeros(n))
                add_est(row, "fdr", false / np.maximum(size, 1))
                add_est(row, "fwer", false > 0)
                add_est(row, "pfer", false)
                curves.append(row)
    summary = []
    for j, method in enumerate(METHODS):
        for s, rule in enumerate(REPORTS):
            tau, size, false, true = report[:, j, s].T
            row = dict(
                m=m,
                method=method,
                report=rule,
                n=n,
                seed=seed,
                k=K,
                horizon=H,
                change=NU,
                mu=MU,
                q=Q,
            )
            add_est(row, "time", tau)
            add_est(row, "fdr", false / np.maximum(size, 1))
            add_est(row, "fwer", false > 0)
            add_est(row, "pfer", false)
            add_est(row, "declarations", size)
            add_est(row, "true_count", true)
            add_est(row, "tdr", true / m if m else np.zeros(n))
            summary.append(row)
    delays = []
    if m:
        for j, method in enumerate(METHODS):
            for f, fraction in enumerate((0.8, 1.0)):
                row = dict(m=m, method=method, fraction=fraction, n=n, seed=seed)
                add_est(row, "restricted_delay", recovery[:, j, f] - NU + 1)
                add_est(row, "nondetection", recovery[:, j, f] > H)
                delays.append(row)
    np.savez_compressed(
        OUT / f"tradeoff_replicates_m{m}.npz",
        report=report,
        recovery=recovery,
        methods=np.asarray(METHODS),
        reports=np.asarray(REPORTS),
        seed=seed,
        n=n,
        k=K,
        horizon=H,
        change=NU,
        mu=MU,
        q=Q,
    )
    return summary, delays, curves


def draw_curve(ax, rows, m, methods):
    for j in methods:
        curve = [r for r in rows if r["m"] == m and r["method"] == METHODS[j]]
        times = np.asarray([r["post_observations"] for r in curve])
        mean = np.asarray([r["tdr"] for r in curve])
        se = np.asarray([r["tdr_se"] for r in curve])
        ax.plot(
            times,
            mean,
            color=COLORS[j],
            lw=2.1,
            label=DISPLAY_LABELS[j],
            ls=("-", "--", "-.", ":")[j],
        )
        ax.fill_between(
            times,
            np.maximum(0, mean - 1.96 * se),
            np.minimum(1, mean + 1.96 * se),
            color=COLORS[j],
            alpha=0.14,
            linewidth=0,
        )
    ax.set(xlim=(0, 180), ylim=(0, 1.025), xlabel="Observations after the change")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(axis="y", color="#e7e7e7", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def figures(summary, curves):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(6.45, 2.8), gridspec_kw={"width_ratios": [1, 1.85]}
    )
    null = [
        next(
            r
            for r in summary
            if r["m"] == 0 and r["method"] == method and r["report"] == "first_1_capped"
        )
        for method in METHODS[:3]
    ]
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        r"(a) False-alarm probability $\downarrow$" + "\nNo changes",
        fontsize=9.5,
        pad=11,
    )
    ax.text(0.5, 0.91, r"Horizon 400; budget $q=0.05$", ha="center", fontsize=9)
    ax.text(0.02, 0.73, "Method", ha="left", fontsize=8.5)
    ax.text(0.74, 0.73, "Prob.", ha="right", fontsize=8.5)
    ax.text(1.0, 0.73, "MC SE", ha="right", fontsize=8.5)
    ax.hlines(0.685, 0, 1, color=".45", linewidth=0.7)
    row_labels = ("e-d-BH\n(ours)", "e-d-Bonferroni\n(ours)", "Per-stream SR")
    for j, (record, label, y) in enumerate(zip(null, row_labels, (0.56, 0.36, 0.16))):
        ax.text(0.02, y, label, color=COLORS[j], va="center", ha="left", fontsize=9)
        ax.text(0.74, y, f"{record['fdr']:.4f}", va="center", ha="right", fontsize=9)
        ax.text(1.0, y, f"{record['fdr_se']:.4f}", va="center", ha="right", fontsize=9)
    ax.hlines(0.065, 0, 1, color=".45", linewidth=0.7)
    draw_curve(axes[1], curves, 50, range(3))
    axes[1].set(
        ylabel="Proportion identified",
        title=r"(b) Identification $\uparrow$" + "\n50 of 100 streams change",
    )
    axes[1].legend(loc="lower right", frameon=False)
    fig.tight_layout(w_pad=1.8)
    fig.savefig(OUT / "tradeoff_main.pdf", bbox_inches="tight")
    fig.savefig(OUT / "tradeoff_main.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(6.45, 3.1), sharey=True)
    for ax, m in zip(axes, (10, 50)):
        draw_curve(ax, curves, m, range(4))
        ax.set(title=f"{m} of {K} streams change\n" + r"Identification $\uparrow$")
    axes[0].set_ylabel("Proportion identified")
    axes[1].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "tradeoff_allmethods.pdf", bbox_inches="tight")
    fig.savefig(OUT / "tradeoff_allmethods.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Redraw saved results without simulations.",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.figures_only:

        def read_numeric(path):
            with path.open() as f:
                return [
                    {
                        key: value if key in ("method", "report") else float(value)
                        for key, value in row.items()
                    }
                    for row in csv.DictReader(f)
                ]

        figures(
            read_numeric(OUT / "tradeoff_summary.csv"),
            read_numeric(OUT / "tradeoff_curves.csv"),
        )
        print("Redrew saved tradeoff figures without rerunning simulations.")
        return
    started = time.time()
    summary, delays, curves = [], [], []
    design = ROOT / "tradeoff_design.txt"
    metadata = dict(
        k=K,
        horizon=H,
        change=NU,
        mu=MU,
        q=Q,
        alpha=Q / H,
        methods=METHODS,
        reports=REPORTS,
        n=args.n,
        changed_counts=[0, 10, 50],
        python=platform.python_version(),
        numpy=np.__version__,
        matplotlib=matplotlib.__version__,
        design_sha256=hashlib.sha256(design.read_bytes()).hexdigest(),
        seeds={str(m): 61006200 + m for m in (0, 10, 50)},
    )
    (OUT / "tradeoff_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for m in (0, 10, 50):
        s, d, c = simulate(m, args.n)
        summary.extend(s)
        delays.extend(d)
        curves.extend(c)
        print(
            f"Completed m={m}, n={args.n}, elapsed={time.time()-started:.1f}s",
            flush=True,
        )
        write_csv(OUT / "tradeoff_summary.csv", summary)
        write_csv(OUT / "tradeoff_delay.csv", delays)
        write_csv(OUT / "tradeoff_curves.csv", curves)
    figures(summary, curves)
    metadata["elapsed_seconds"] = time.time() - started
    (OUT / "tradeoff_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved comparison to {OUT}; elapsed={time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
