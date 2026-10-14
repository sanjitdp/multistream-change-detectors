#!/usr/bin/env python3
"""Generate the compact manuscript table from saved Gaussian tradeoff results."""
import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results"
rows = list(csv.DictReader((OUT / "tradeoff_summary.csv").open()))
delays = list(csv.DictReader((OUT / "tradeoff_delay.csv").open()))


def cell(x, key, d):
    return f"${float(x[key]):.{d}f}\\;({float(x[key+'_se']):.{d}f})$"


text = r"""\begin{table}[!htbp]
\centering\small
\caption{Gaussian comparison with $K=100$, $H=400$ and $q=0.05$, using 4000 paired runs per setting. Changes have size $0.5$ and occur at time 101. Null alarm probability is the probability of any declaration by $H$ when no stream changes. At descending rank $r$, e-d-BH uses the SR threshold $800{,}000/r$; e-d-Bonferroni and the separate SR charts use $800{,}000$ and $8000$, respectively. SHEREF uses $2000t/r$ at time $t$. Identification columns give the mean number of post-change observations needed to declare 80\% of the changed streams. Parentheses give Monte Carlo standard errors.}
\label{tab:tradeoff}
\begin{tabular}{lrrr}
\toprule
 & Null alarm & \multicolumn{2}{c}{Identification delay $\downarrow$} \\
Method & probability $\downarrow$ & 10 changes & 50 changes \\
\midrule
"""
for method, label in [
    ("e-d-BH (ours)", "e-d-BH (ours)"),
    ("Parallel SR", "e-d-Bonferroni (ours)"),
    ("Per-stream SR", "Per-stream SR"),
    ("SR/t e-BH (SHEREF)", "SHEREF"),
]:
    null = next(
        r
        for r in rows
        if int(r["m"]) == 0
        and r["method"] == method
        and r["report"] == "first_1_capped"
    )
    vals = [cell(null, "fdr", 3)]
    for m in [10, 50]:
        row = next(
            r
            for r in delays
            if int(r["m"]) == m
            and r["method"] == method
            and float(r["fraction"]) == 0.8
        )
        vals.append(cell(row, "restricted_delay", 1))
    text += label + " & " + " & ".join(vals) + r" \\" + "\n"
text += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
(OUT / "tradeoff_table.tex").write_text(text)
