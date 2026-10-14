"""Tables and figures from the saved external identification results."""

import argparse
import csv
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/external_identification_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/external_identification_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
METHODS = ("BH", "Bonf", "GL", "GL_time", "MD", "LFDR")
NAMES = (
    "e-d-BH (ours)",
    "e-d-Bonferroni (ours)",
    "Gandy--Lau",
    "Gandy--Lau (time adj.)",
    "MD-FDR",
    "Posterior LFDR",
)
PLOT_NAMES = (
    "e-d-BH (ours)",
    "e-d-Bonferroni (ours)",
    "Gandy–Lau (fixed-time)",
    "Gandy–Lau (time-adjusted)",
)
COLORS = ("#0072B2", "#D55E00", "#009E73", "#8B5BA5")
STYLES = ("-", "--", "-.", ":")


def read(name):
    with (OUT / name).open() as f:
        return [
            {k: v if k in ("case", "method") else float(v) for k, v in r.items()}
            for r in csv.DictReader(f)
        ]


def fetch(rows, case, method):
    return next(r for r in rows if r["case"] == case and r["method"] == method)


def number(x, digits, omit_zero=False):
    value = f"{x:.{digits}f}"
    return value[1:] if omit_zero and value.startswith("0.") else value


def cell(row, key, digits=4, omit_zero=True):
    value, se = row[key], row[key + "_se"]
    if value == 0 and se == 0:
        return r"$0^{a}$"
    return number(value, digits, omit_zero) + " (" + number(se, digits, True) + ")"


def begin(caption, label, columns, placement="!htbp"):
    return [
        r"\begin{table}[" + placement + "]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begingroup\fontsize{9}{11}\selectfont",
        r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{@{}" + columns + "@{}}",
        r"\hline",
    ]


def finish(lines, name):
    lines.extend([r"\hline", r"\end{tabular}", r"\endgroup", r"\end{table}"])
    (OUT / name).write_text("\n".join(lines) + "\n")


ZERO_NOTE = (
    r"A superscript $a$ means zero events observed, with empirical MC SE zero; "
    r"it does not assert a zero error probability."
)
THRESHOLD_NOTE = (
    r"With $K=100$ and $\alpha=0.05/400$, our SR procedures use "
    r"$M_t^{[r]}\geq800{,}000/r$ for e-d-BH (rank $r$) and "
    r"$M_t^{(k)}\geq800{,}000$ for e-d-Bonferroni."
)


def tables(rows):
    caption = (
        r"Error and identification at the first nonempty report through time 400 "
        r"(an empty set if there is no report). There are 100 streams; in the two alternative "
        r"settings, 50 change by a mean of 0.5 at the indicated time. True counts are unconditional "
        r"and are zero for pre-change or absent reports. Entries are estimates (Monte Carlo standard "
        r"errors) from 3000 paired runs. Smaller FDR $\downarrow$ and larger true count $\uparrow$ "
        r"are preferable. The methods have different stated guarantees; see the text. "
        + THRESHOLD_NOTE
    )
    lines = begin(caption, "tab:external-main", "lrrrrr")
    lines.extend(
        [
            r" & No changes & \multicolumn{2}{c}{Change at 101} & \multicolumn{2}{c}{Change at 251} \\",
            r"Method & FDR $\downarrow$ & FDR $\downarrow$ & True count $\uparrow$ & FDR $\downarrow$ & True count $\uparrow$ \\",
            r"\hline",
        ]
    )
    for method, name in zip(METHODS, NAMES):
        null, early, late = (
            fetch(rows, case, method) for case in ("null", "m50_n101", "m50_n251")
        )
        values = [
            cell(null, "fdr"),
            cell(early, "fdr"),
            cell(early, "true_count", 3, False),
            cell(late, "fdr"),
            cell(late, "true_count", 3, False),
        ]
        lines.append(name + " & " + " & ".join(values) + r" \\")
    finish(lines, "external_main_table.tex")

    caption = (
        r"First reports when 10 of 100 streams change. FDR, true count and the probability "
        r"of a pre-change report are unconditional. Delay counts observations from the change to "
        r"a report, conditional on no pre-change report; it need not be a true discovery. All runs "
        r"reported by time 400, so the no-report probability and its empirical MC SE are zero "
        r"for every row. Entries are estimates (Monte Carlo standard errors) from 3000 runs. "
        r"Conditional sample sizes and uncapped-delay details are retained in the accompanying CSV. "
        + THRESHOLD_NOTE
    )
    lines = begin(caption, "tab:external-sparse", "lrrrr", placement="H")
    lines.extend(
        [
            r"Method & FDR $\downarrow$ & True count $\uparrow$ & Delay $\downarrow$ & Pre-change $\downarrow$ \\",
            r"\hline",
        ]
    )
    for nu in (101, 251):
        lines.append(r"\multicolumn{5}{c}{Change at " + str(nu) + r"} \\")
        for method, name in zip(METHODS, NAMES):
            r = fetch(rows, f"m10_n{nu}", method)
            values = [
                cell(r, "fdr"),
                cell(r, "true_count", 3, False),
                cell(r, "restricted_delay", 2, False),
                cell(r, "prechange_report"),
            ]
            lines.append(name + " & " + " & ".join(values) + r" \\")
        if nu == 101:
            lines.append(r"\hline")
    finish(lines, "external_sparse_table.tex")

    caption = (
        r"First reports under the correctly specified changepoint prior. Each stream "
        r"independently has no change with probability $1/2$; otherwise its first post-change "
        r"observation is geometric with hazard 0.01. The Bayesian FDR guarantees average over "
        r"this prior. Delay is from the first actual change in the family to a report, conditional "
        r"on no report before that change. It is not a delay to a true discovery. Every run "
        r"reported by time 400. Entries are estimates (Monte Carlo standard errors), with 3000 "
        r"runs. " + THRESHOLD_NOTE + " " + ZERO_NOTE
    )
    lines = begin(caption, "tab:external-prior", "lrrrr", placement="H")
    lines.extend(
        [
            r"Method & FDR $\downarrow$ & True count $\uparrow$ & Delay $\downarrow$ & Pre-change $\downarrow$ \\",
            r"\hline",
        ]
    )
    for method, name in zip(METHODS, NAMES):
        r = fetch(rows, "prior", method)
        values = [
            cell(r, "fdr"),
            cell(r, "true_count", 3, False),
            cell(r, "restricted_delay", 2, False),
            cell(r, "prechange_report"),
        ]
        lines.append(name + " & " + " & ".join(values) + r" \\")
    finish(lines, "external_prior_table.tex")

    caption = (
        r"Reporting delays and pre-change reports for the 50-change settings in "
        r"Table~\ref{tab:external-main}. Delay counts observations from the change to a report, "
        r"conditional on no pre-change report; it need not be a true discovery. Pre-change-report "
        r"probabilities are unconditional, so the methods condition on different sets of runs. "
        r"No reports were censored at 400 in these settings. Entries are estimates (Monte Carlo "
        r"standard errors) from 3000 runs; conditioning counts are in the accompanying CSV. "
        + THRESHOLD_NOTE
    )
    lines = begin(caption, "tab:external-delays", "lrrrr", placement="H")
    lines.extend(
        [
            r" & \multicolumn{2}{c}{Change at 101} & \multicolumn{2}{c}{Change at 251} \\",
            r"Method & Delay $\downarrow$ & Pre-change $\downarrow$ & Delay $\downarrow$ & Pre-change $\downarrow$ \\",
            r"\hline",
        ]
    )
    for method, name in zip(METHODS, NAMES):
        early, late = (fetch(rows, case, method) for case in ("m50_n101", "m50_n251"))
        values = [
            cell(early, "restricted_delay", 2, False),
            cell(early, "prechange_report"),
            cell(late, "restricted_delay", 2, False),
            cell(late, "prechange_report"),
        ]
        lines.append(name + " & " + " & ".join(values) + r" \\")
    finish(lines, "external_delay_table.tex")


def setup_plot():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def curve(ax, rows, case, metric):
    for j, method in enumerate(METHODS[:4]):
        records = [r for r in rows if r["case"] == case and r["method"] == method]
        x = np.array([r["t"] for r in records])
        y = np.array([r[metric] for r in records])
        e = 1.96 * np.array([r[metric + "_se"] for r in records])
        ax.plot(x, y, color=COLORS[j], ls=STYLES[j], lw=1.4, label=PLOT_NAMES[j])
        ax.fill_between(
            x,
            np.maximum(0, y - e),
            np.minimum(1, y + e),
            color=COLORS[j],
            alpha=0.10,
            linewidth=0,
        )
    if metric == "fdr":
        ax.axhline(0.05, color=".55", lw=0.7, ls=(0, (2, 3)), zorder=0)
    ax.grid(axis="y", alpha=0.15, lw=0.5)
    ax.set_xlim(0, 400)
    ax.set_xticks([0, 100, 200, 300, 400])


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=240)
    plt.close(fig)


def figures(rows):
    setup_plot()
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.05), sharex=True, sharey="row")
    for col, nu in enumerate((101, 251)):
        for row, metric in enumerate(("fdr", "tdr")):
            ax = axes[row, col]
            curve(ax, rows, f"m50_n{nu}", metric)
            ax.axvline(nu, color=".55", lw=0.8, ls=":")
            ax.set_ylim(0, 0.075 if metric == "fdr" else 1.025)
            if row == 0:
                ax.set_title(
                    f"50 changes at {nu}\n" + r"False discovery rate $\downarrow$",
                    pad=8,
                )
            else:
                ax.set_title(r"True discovery rate $\uparrow$", pad=7)
                ax.set_xlabel("Time $t$")
            if col == 0:
                ax.set_ylabel("FDR at time $t$" if row == 0 else "TDR at time $t$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handlelength=2.4,
        columnspacing=1.8,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92], h_pad=1.15, w_pad=1.5)
    save(fig, "external_fixedtime")


def fixedtime_caption():
    (OUT / "external_fixedtime_captions.tex").write_text(
        r"""% Suggested appendix figure captions; q=.05 and alpha=q/400.
\begin{figure}[tb]
\centering
\includegraphics[width=\linewidth]{experiments/results/external_fixedtime.pdf}
\caption{Fixed-time error and identification when 50 of 100 independent Gaussian streams change by a mean of 0.5 at time 101 or 251. FDR and TDR concern the current reported set at the displayed time, without stopping at the first report or accumulating past declarations. Native Gandy--Lau uses fixed-time level 0.05; its time-adjusted version uses $0.05/400$. """
        + THRESHOLD_NOTE
        + r""" The horizontal reference is 0.05. Shading gives $1.96$ Monte Carlo standard errors from 3000 paired runs; vertical lines mark changes.}
\label{fig:external-fixedtime}
\end{figure}
"""
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="write tables and the suggested caption without regenerating plots",
    )
    args = parser.parse_args()
    tables(read("external_identification_summary.csv"))
    fixedtime_caption()
    if not args.tables_only:
        figures(read("external_identification_curves.csv"))
    print(
        "Wrote external tables and suggested caption from saved CSVs; no simulations."
        if args.tables_only
        else "Wrote external tables and fixed-time figures from saved CSVs; no simulations."
    )
