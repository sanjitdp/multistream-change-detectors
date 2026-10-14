#!/usr/bin/env python3
"""Create the manuscript figures and tables from saved simulation output."""
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(name):
    rows = list(csv.DictReader((OUT / name).open()))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (ValueError, TypeError):
                pass
    return rows


SUMMARY = read("summary.csv")
CURVES = read("curves.csv")
ARLS = read("arl_validation.csv")


def get(rows=SUMMARY, **conditions):
    result = [r for r in rows if all(r[k] == v for k, v in conditions.items())]
    assert len(result) == 1, (conditions, len(result))
    return result[0]


def cell(row, name, scale=1.0, digits=1):
    if 0 < row[name + "_se"] * scale < 0.5 * 10 ** (-digits):
        digits += 1
    return f"${row[name]*scale:.{digits}f}\\;({row[name+'_se']*scale:.{digits}f})$"


STYLE = (
    ("BH", "theory", "e-d-BH (ours)", "#0072B2", "-"),
    ("SR-max", "theory", "e-d-Bonferroni/SR (ours)", "#D55E00", "--"),
    ("SR/t BH", "fixed-time", "SHEREF (fixed-time FDR)", "#009E73", "-."),
)
METHOD_LABELS = {
    "BH": "e-d-BH (ours)",
    "SR-max": "e-d-Bonferroni/SR (ours)",
    "CUSUM-max": "e-d-Bonferroni/CUSUM (ours)",
    "Sum-CUSUM": "Sum-CUSUM (global alarm only)",
}
METHOD_COLORS = {
    "BH": "#0072B2",
    "SR-max": "#D55E00",
    "CUSUM-max": "#4D4D4D",
    "Sum-CUSUM": "#8C56A2",
}
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def curve(ax, metric, mu, rho, scenario, **kwargs):
    for method, calibration, label, color, linestyle in STYLE:
        data = [
            r
            for r in CURVES
            if r["mu"] == mu
            and r["rho"] == rho
            and r["scenario"] == scenario
            and r["method"] == method
            and r["calibration"] == calibration
            and r.get(metric) != ""
        ]
        x = np.array([r["t"] for r in data])
        y = np.array([r[metric] for r in data])
        e = np.array([r[metric + "_se"] for r in data])
        ax.plot(x, y, label=label, color=color, linestyle=linestyle, lw=1.4)
        ax.fill_between(
            x,
            np.maximum(0, y - 1.96 * e),
            np.minimum(1, y + 1.96 * e),
            color=color,
            alpha=0.11,
            linewidth=0,
        )
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.set_xlim(kwargs.get("xmin", 0), kwargs.get("xmax", 400))
    if scenario != "null":
        ax.axvline(101, color=".55", lw=0.7, linestyle=":")


def figures():
    fig, axes = plt.subplots(2, 2, figsize=(6.45, 4.25), sharex=True)
    for col, rho in enumerate((0.0, 0.5)):
        for row, scenario in enumerate(("null", "dense")):
            ax = axes[row, col]
            curve(ax, "fdr", 0.4, rho, scenario)
            ax.set_title(
                ("Global null" if row == 0 else "25 changes at time 101")
                + rf", $\rho={rho:g}$"
                + "\n"
                + r"FDR $\downarrow$"
            )
            if col == 0:
                ax.set_ylabel(r"FDR at time $t$")
            if row == 1:
                ax.set_xlabel("Time $t$")
            ax.set_ylim(0, 0.026 if row == 0 else 0.018)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
        handlelength=2,
        columnspacing=1,
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955], h_pad=1.1, w_pad=1.3)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"type1.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(6.45, 4.25), sharex="row", sharey=True)
    for row, mu in enumerate((0.4, 0.75)):
        xmax = 300 if mu == 0.4 else 200
        for col, (scenario, metric, title) in enumerate(
            (
                ("sparse", "tdr", "5 changed: true discovery rate"),
                ("dense", "tdr", "25 changed: true discovery rate"),
                ("dense", "proball", "25 changed: all identified"),
            )
        ):
            ax = axes[row, col]
            curve(ax, metric, mu, 0.0, scenario, xmin=100, xmax=xmax)
            ax.set_ylim(0, 1.03)
            if row == 0:
                ax.set_title(title + r" $\uparrow$", fontsize=8.2)
            if col == 0:
                ax.set_ylabel(rf"$\mu={mu:g}$" + "\nProportion / probability")
            if row == 1:
                ax.set_xlabel("Time $t$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
        handlelength=2,
        columnspacing=1,
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955], h_pad=1.1, w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"power.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_figure(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def error_ratio_figure():
    """Plot error-to-patience ratios at the selected stopping times."""
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(6.45, 4.35), sharex=True, sharey=True)
    scenarios = (
        ("null", "Global null"),
        ("sparse", "5 changes"),
        ("dense", "25 changes"),
        ("staggered", "Staggered changes"),
    )
    y = np.arange(len(scenarios))
    for row, (prefix, stopping_label) in enumerate(
        (("bounded_first", "First declaration"), ("bounded_five", "Five declarations"))
    ):
        for col, rho in enumerate((0.0, 0.5)):
            ax = axes[row, col]
            for method, metric, offset, marker in (
                ("BH", "fdr", -0.12, "o"),
                ("SR-max", "pfer", 0.12, "s"),
            ):
                records = [
                    get(
                        mu=0.4,
                        rho=rho,
                        scenario=scenario,
                        method=method,
                        calibration="theory",
                    )
                    for scenario, _ in scenarios
                ]
                key = f"{prefix}_{metric}_ratio"
                means = np.array([r[key] for r in records]) * 1000
                errors = np.array([r[key + "_se"] for r in records]) * 1000 * 1.96
                ax.errorbar(
                    means,
                    y + offset,
                    xerr=errors,
                    fmt=marker,
                    color=METHOD_COLORS[method],
                    markersize=4.2,
                    capsize=2.2,
                    elinewidth=1.0,
                    markeredgewidth=0.8,
                )
            ax.axvline(1.0, color=".4", linestyle="--", linewidth=1.0)
            if row == 0:
                ax.set_title(
                    ("Independent streams" if rho == 0 else "Correlated streams")
                    + rf" ($\rho={rho:g}$)",
                    fontsize=9.5,
                    pad=10,
                )
            if row == 0 and col == 1:
                ax.text(
                    1.0,
                    -0.72,
                    "Bound",
                    ha="center",
                    va="bottom",
                    color=".35",
                    fontsize=8,
                )
            if col == 0:
                ax.set_ylabel(stopping_label, fontsize=9.5, labelpad=7)
                ax.set_yticks(y, [label for _, label in scenarios])
            ax.set_xlim(-0.025, 1.065)
            ax.set_ylim(3.5, -0.5)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
            ax.tick_params(axis="y", length=0, labelsize=8.5)
            ax.grid(axis="x", color=".91", linewidth=0.6)
            ax.set_axisbelow(True)
            if row == 1:
                ax.set_xlabel(
                    r"Error / patience, relative to bound $\downarrow$", fontsize=9
                )
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS["BH"],
            marker="o",
            lw=1,
            label="e-d-BH (ours): FDR",
        ),
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS["SR-max"],
            marker="s",
            lw=1,
            label="e-d-Bonferroni/SR (ours): PFER",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=2,
        frameon=False,
        fontsize=9,
        handlelength=1.5,
        columnspacing=1.6,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965], w_pad=1.5, h_pad=1.8)
    save_figure(fig, "error_ratios")


def baseline_delay_figure():
    """Plot global detection and stream identification delays."""
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(6.45, 4.65), sharey=True)
    methods = ("BH", "SR-max", "CUSUM-max", "Sum-CUSUM")
    ylabels = (
        "e-d-BH (ours)",
        "e-d-Bonferroni\n(ours; SR)",
        "e-d-Bonferroni\n(ours; CUSUM)",
        "Sum-CUSUM\n(global alarm only)",
    )
    for row, mu in enumerate((0.4, 0.75)):
        for col in (0, 1):
            ax = axes[row, col]
            for y, method in enumerate(methods):
                if col == 1 and method == "Sum-CUSUM":
                    ax.text(
                        0.04,
                        y,
                        "No stream-level declarations",
                        transform=ax.get_yaxis_transform(),
                        fontsize=8,
                        color=".4",
                        va="center",
                    )
                    continue
                specs = (
                    (
                        ("sparse", "global_conditional_restricted_delay"),
                        ("dense", "global_conditional_restricted_delay"),
                    )
                    if col == 0
                    else (
                        ("dense", "80_restricted_delay"),
                        ("dense", "all_restricted_delay"),
                    )
                )
                for j, (scenario, metric) in enumerate(specs):
                    record = get(
                        mu=mu,
                        rho=0.0,
                        scenario=scenario,
                        method=method,
                        calibration="matched",
                    )
                    ax.errorbar(
                        record[metric],
                        y + (-0.13 if j == 0 else 0.13),
                        xerr=1.96 * record[metric + "_se"],
                        fmt="o" if j == 0 else "D",
                        color=METHOD_COLORS[method],
                        markerfacecolor="white" if j == 0 else METHOD_COLORS[method],
                        markersize=4.5,
                        capsize=2.3,
                        elinewidth=1.1,
                        markeredgewidth=1.0,
                    )
            ax.set_ylim(3.55, -0.55)
            ax.tick_params(axis="y", length=0, labelsize=8.5)
            ax.set_yticks(np.arange(4), ylabels)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color=".91", linewidth=0.6)
            if col == 0:
                ax.set_xlim(0, 52 if mu == 0.4 else 21)
                ax.set_xticks(
                    [0, 10, 20, 30, 40, 50] if mu == 0.4 else [0, 5, 10, 15, 20]
                )
                ax.set_ylabel(rf"Mean shift $\mu={mu:g}$", labelpad=8, fontsize=9.5)
            else:
                ax.set_xlim(0, 224 if mu == 0.4 else 74)
                ax.set_xticks([0, 50, 100, 150, 200] if mu == 0.4 else [0, 20, 40, 60])
            if row == 0:
                ax.set_title(
                    (
                        r"First global alarm $\downarrow$"
                        if col == 0
                        else r"Identification delay $\downarrow$"
                    ),
                    fontsize=9.5,
                    pad=32,
                )
                metric_labels = (
                    ("5 changed", "25 changed") if col == 0 else ("80% of 25", "All 25")
                )
                handles = [
                    Line2D(
                        [0],
                        [0],
                        linestyle="none",
                        marker="o",
                        color=".25",
                        markerfacecolor="white",
                        markersize=4.5,
                        label=metric_labels[0],
                    ),
                    Line2D(
                        [0],
                        [0],
                        linestyle="none",
                        marker="D",
                        color=".25",
                        markersize=4.5,
                        label=metric_labels[1],
                    ),
                ]
                ax.legend(
                    handles=handles,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.005),
                    ncol=2,
                    frameon=False,
                    fontsize=8,
                    handletextpad=0.25,
                    columnspacing=0.85,
                    borderaxespad=0,
                )
            if row == 1:
                ax.set_xlabel("Mean restricted delay (observations)", fontsize=8.8)
    fig.tight_layout(rect=[0, 0, 1, 1], w_pad=2.0, h_pad=1.5)
    save_figure(fig, "baseline_delays")


def tables():
    lines = [
        r"\begin{table}[tb]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{2.25pt}",
        r"\caption{Numerical values for the selected-stopping-rule error comparison, with $\mu=0.4$ and level $10^{-3}$. Entries are $10^3$ times the estimated error/patience ratio, with Monte Carlo standard errors in parentheses. $\tau_r$ is the first time at least $r$ streams are declared, capped at 400. e-d-BH (ours) uses FDR; Parallel SR (e-d-Bonferroni) uses PFER. Each corresponding theoretical upper bound is 1. These are selected stopping rules, not estimates of the EOP supremum.}",
        r"\label{tab:simulation-errors}",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r" & & \multicolumn{2}{c}{$\tau_1$} & \multicolumn{2}{c}{$\tau_5$} \\",
        r"$\rho$ & Configuration & e-d-BH (ours) & Parallel SR & e-d-BH (ours) & Parallel SR \\",
        r"\hline",
    ]
    for rho in (0.0, 0.5):
        for scenario, label in (
            ("null", "Global null"),
            ("sparse", "5 changes"),
            ("dense", "25 changes"),
            ("staggered", "Staggered"),
        ):
            bh = get(
                mu=0.4, rho=rho, scenario=scenario, method="BH", calibration="theory"
            )
            bonf = get(
                mu=0.4,
                rho=rho,
                scenario=scenario,
                method="SR-max",
                calibration="theory",
            )
            cells = [
                cell(bh, "bounded_first_fdr_ratio", 1000, 3),
                cell(bonf, "bounded_first_pfer_ratio", 1000, 3),
                cell(bh, "bounded_five_fdr_ratio", 1000, 3),
                cell(bonf, "bounded_five_pfer_ratio", 1000, 3),
            ]
            lines.append(f"{rho:g} & {label} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUT / "error_table.tex").write_text("\n".join(lines) + "\n")

    lines = [
        r"\begin{table}[tb]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Numerical values for the baseline comparison after empirical calibration to global-null ARL 1000 under independence. ARLs use 1500 fresh, uncensored null runs. Delay columns use 2000 runs with $m$ changes at 101; global alarm delays condition on no earlier alarm. Identification columns concern $m=25$. All delays are restricted at 301, including nondetections; Monte Carlo standard errors are in parentheses. Our e-d-Bonferroni procedure is shown with both SR and CUSUM inputs. Sum-CUSUM makes only a global declaration.}",
        r"\label{tab:simulation-delays}",
        r"\begin{tabular}{llrrr}",
        r"\multicolumn{5}{l}{(a) First global alarm} \\[3pt]",
        r"\hline",
        r"$\mu$ & Method & ARL & $m=5$ $\downarrow$ & $m=25$ $\downarrow$ \\",
        r"\hline",
    ]
    for mu in (0.4, 0.75):
        for method in ("BH", "SR-max", "CUSUM-max", "Sum-CUSUM"):
            arl = get(rows=ARLS, mu=mu, method=method, calibration="matched")
            sparse = get(
                mu=mu, rho=0.0, scenario="sparse", method=method, calibration="matched"
            )
            dense = get(
                mu=mu, rho=0.0, scenario="dense", method=method, calibration="matched"
            )
            cells = [
                cell(arl, "arl", digits=0),
                cell(sparse, "global_conditional_restricted_delay"),
                cell(dense, "global_conditional_restricted_delay"),
            ]
            label = METHOD_LABELS[method]
            lines.append(f"{mu:g} & {label} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\par\medskip",
        r"\begin{tabular}{llrr}",
        r"\multicolumn{4}{l}{(b) Identification of the 25 changed streams} \\[3pt]",
        r"\hline",
        r"$\mu$ & Method & 80\% $\downarrow$ & All $\downarrow$ \\",
        r"\hline",
    ]
    for mu in (0.4, 0.75):
        for method in ("BH", "SR-max", "CUSUM-max"):
            dense = get(
                mu=mu, rho=0.0, scenario="dense", method=method, calibration="matched"
            )
            cells = [
                cell(dense, x) for x in ("80_restricted_delay", "all_restricted_delay")
            ]
            lines.append(
                f"{mu:g} & {METHOD_LABELS[method]} & " + " & ".join(cells) + r" \\"
            )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUT / "delay_table.tex").write_text("\n".join(lines) + "\n")


def fragment():
    bh = get(mu=0.4, rho=0.0, scenario="dense", method="BH", calibration="theory")
    bonf = get(mu=0.4, rho=0.0, scenario="dense", method="SR-max", calibration="theory")
    fixed = get(
        mu=0.4, rho=0.0, scenario="dense", method="SR/t BH", calibration="fixed-time"
    )
    stagbh = get(
        mu=0.4, rho=0.0, scenario="staggered", method="BH", calibration="theory"
    )
    stagbonf = get(
        mu=0.4, rho=0.0, scenario="staggered", method="SR-max", calibration="theory"
    )
    sparse = get(mu=0.4, rho=0.0, scenario="sparse", method="BH", calibration="theory")
    null = get(mu=0.4, rho=0.0, scenario="null", method="BH", calibration="theory")
    text = r"""% Generated from actual simulations by experiments/summarize.py.
% Figure and input paths below assume compilation from the repository root.
\section{Simulations}
\label{sec:simulations}

We compare false discoveries, detection delay and identification in a Gaussian mean-change problem. There are $K=50$ streams, with observations distributed as $\mathcal N(0,1)$ before a change and $\mathcal N(\mu,1)$ afterwards, where $\mu\in\{0.4,0.75\}$. Observation vectors are independent over time, with cross-stream equicorrelation $\rho\in\{0,0.5\}$. We consider the global null, changes in either 5 or 25 streams at time 101, and staggered changes in 10 streams at 101 and another 15 at 151. All other streams remain unchanged. Each configuration uses 2000 independent runs, ending at time 400.

The SR statistic is $M_t^{(k)}=(M_{t-1}^{(k)}+1)\exp\{\mu X_t^{(k)}-\mu^2/2\}$, with $M_0^{(k)}=0$. We first use e-d-BH and e-d-Bonferroni at their theoretical EOP level $10^{-3}$. For these inputs, e-d-Bonferroni is exactly a collection of parallel SR charts with threshold $K/\alpha$. We also apply e-BH to $M_t^{(k)}/t$ at level 0.05. This is the fixed-sensor, unboosted SHEREF rule of \citet{lee2026sequential}, with declarations indexed by the most recent observation. Its target is fixed-time FDR, so this comparison illustrates the effect of different error requirements.

\subsection{False discoveries}

Figure~\ref{fig:simulation-errors} plots FDR at deterministic times under the global null and a partial null. These probabilities need not stay below $10^{-3}$ for the constant-threshold rules: that is an EOP level, not a fixed-time FDR level. For example, with $\mu=0.4$ and independent streams, e-d-BH has estimated FDR @NULLFDR@ at time 200 under the global null, and @DENSEFDR@ when 25 streams change. In the latter configuration, e-d-Bonferroni has estimated PFER @DENSEPFER@.

We also evaluate error divided by expected stopping time at two specified rules: the first time at least one or five streams are declared, capped at 400. Table~\ref{tab:simulation-errors} gives these ratios for $\mu=0.4$. The estimates are consistent with the respective FDR and PFER guarantees, including with correlated streams and staggered changes. A finite collection of stopping rules cannot estimate or verify the supremum defining EOP.

\begin{figure}[tb]
\centering
\includegraphics[width=\textwidth]{experiments/results/type1.pdf}
\caption{FDR at deterministic times, with $\mu=0.4$. e-d-BH and e-d-Bonferroni use EOP level $10^{-3}$; SR/t BH uses fixed-time FDR level 0.05. Shading shows pointwise Monte Carlo uncertainty of 1.96 standard errors. The vertical line marks time 101 in the partial-null configurations.}
\label{fig:simulation-errors}
\end{figure}

\input{experiments/results/error_table.tex}

\subsection{Detection and identification}

Figure~\ref{fig:simulation-power} reports the expected proportion of changed streams currently declared and the probability that all changed streams are declared at the same time. The complement of the former is the expected proportion of changed streams missed. With 25 independent changes and $\mu=0.4$, e-d-BH's true discovery rate at time 150 is @BHTDR@, compared with @BONFTDR@ for e-d-Bonferroni and @FIXEDTDR@ for SR/t BH. With only five changed streams, the corresponding value for e-d-BH is @SPARSETDR@. The benefit of the step-up threshold is larger when more streams change.

For delay, the first post-change observation counts as one. We record the first time the current declaration set contains 80\% or all of the changed streams. A run that fails to reach the target by time 400 is assigned time 401, and is retained in the mean. These are restricted means, not estimates of an uncensored expected delay. In the independent, 25-stream, $\mu=0.4$ configuration, the mean restricted times to 80\% identification are @BH80@ and @BONF80@ for e-d-BH and e-d-Bonferroni; the corresponding times to all identification are @BHALL@ and @BONFALL@. The fractions failing to identify all streams by 400 are @BHFAIL@ and @BONFFAIL@. For staggered changes those fractions are @STAGBHFAIL@ and @STAGBONFFAIL@. Thus the differences persist when changes occur at different times.

\begin{figure}[tb]
\centering
\includegraphics[width=\textwidth]{experiments/results/power.pdf}
\caption{True discovery rates and simultaneous identification probabilities under independence. All changed streams change at time 101. The error targets and shading are as in Figure~\ref{fig:simulation-errors}. The two rows use different horizontal scales.}
\label{fig:simulation-power}
\end{figure}

For comparison with classical monitoring rules, we separately calibrate e-d-BH, parallel SR, parallel CUSUM and Sum-CUSUM to global-null ARL 1000 under independence. Write $W_t^{(k)}=\max\{0,W_{t-1}^{(k)}+\mu X_t^{(k)}-\mu^2/2\}$. Parallel CUSUM uses $\max_k W_t^{(k)}$, while Sum-CUSUM uses $\sum_k W_t^{(k)}$, as in \citet{mei2010efficient}. Sum-CUSUM makes only a global declaration, so we evaluate its global detection delay without assigning it stream-level discoveries. Calibration uses 800 null runs, and ARL validation uses 1500 fresh runs continued until every alarm occurs. Table~\ref{tab:simulation-delays} reports the resulting ARLs; their departures from 1000 reflect calibration and simulation error. These fitted thresholds do not retain the nominal EOP level $10^{-3}$.

\input{experiments/results/delay_table.tex}

Global detection delays in Table~\ref{tab:simulation-delays} condition on no alarm before the change and retain post-change nondetections at the horizon. Sum-CUSUM detects dense changes sooner than the streamwise procedures. e-d-BH and the parallel charts have similar first-alarm delays, while e-d-BH identifies a large fraction of the changed streams sooner. This illustrates the distinction between detecting that something changed and identifying the changed streams with a specified false-discovery guarantee. These experiments concern known positive Gaussian shifts; they do not establish uniform power advantages or cover the full range of scan, mixture and Bayesian procedures. Code, calibration details, Monte Carlo standard errors and results for all configurations accompany the paper.
"""
    replacements = {
        "NULLFDR": f"{null['fdr_200']:.4f}",
        "DENSEFDR": f"{bh['fdr_200']:.4f}",
        "DENSEPFER": f"{bonf['pfer_200']:.4f}",
        "BHTDR": f"{bh['tdr_150']:.3f}",
        "BONFTDR": f"{bonf['tdr_150']:.3f}",
        "FIXEDTDR": f"{fixed['tdr_150']:.3f}",
        "SPARSETDR": f"{sparse['tdr_150']:.3f}",
        "BH80": f"{bh['80_restricted_delay']:.1f}",
        "BONF80": f"{bonf['80_restricted_delay']:.1f}",
        "BHALL": f"{bh['all_restricted_delay']:.1f}",
        "BONFALL": f"{bonf['all_restricted_delay']:.1f}",
        "BHFAIL": f"{bh['all_nondetection']:.4f}",
        "BONFFAIL": f"{bonf['all_nondetection']:.4f}",
        "STAGBHFAIL": f"{stagbh['all_nondetection']:.4f}",
        "STAGBONFFAIL": f"{stagbonf['all_nondetection']:.3f}",
    }
    for key, value in replacements.items():
        text = text.replace("@" + key + "@", value)
    assert "@" not in text
    (OUT / "results.tex").write_text(text)


if __name__ == "__main__":
    figures()
    error_ratio_figure()
    baseline_delay_figure()
    tables()
    fragment()
    print("Created four figures, two tables, and results.tex")
