#!/usr/bin/env python3
"""Exploratory UCI gas-sensor analysis from home_sensor_design.txt."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "home-sensors"
OUT = ROOT / "results"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib_home_sensor"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

K, TRAIN, H, PRE, Q = 8, 360, 420, 360, 0.05
METHODS = ("e-d-BH (ours)", "Parallel SR", "Per-stream SR")
COLORS = ("#2166ac", "#d97918", "#a84266")
EXPECTED_SHA = "7c143b9f4402a8205ebe8072c2ce0967c25741f1865e23d650e981053294f395"
TIMES = np.arange(-1795, 301, 5)


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def acf1(values):
    a, b = values[:-1], values[1:]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_source():
    source = DATA / "source.zip"
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA:
        raise RuntimeError("Source hash differs from the prespecified source.")
    with zipfile.ZipFile(source) as outer:
        meta_text = outer.read("HT_Sensor_metadata.dat").decode()
        metadata = []
        for line in meta_text.splitlines()[1:]:
            if not line.strip():
                continue
            ident, date, category, start, duration = line.split()
            date = datetime.strptime(date, "%m-%d-%y").date().isoformat()
            metadata.append(
                dict(
                    id=int(ident),
                    date=date,
                    category=category,
                    onset_clock_hours=float(start),
                    duration_hours=float(duration),
                )
            )
        selected = [r for r in metadata if r["date"] >= "2015-08-20"]
        with zipfile.ZipFile(io.BytesIO(outer.read("HT_Sensor_dataset.zip"))) as inner:
            with inner.open("HT_Sensor_dataset.dat") as f:
                raw = np.loadtxt(f, skiprows=1)
    assert (
        len(selected) == 47
        and sum(r["category"] != "background" for r in selected) == 29
    )
    return metadata, selected, raw


def bin_record(rows):
    seconds = rows[:, 1] * 3600
    resistances = rows[:, 2:10]
    valid = (
        np.isfinite(seconds)
        & np.isfinite(resistances).all(axis=1)
        & (resistances > 0).all(axis=1)
        & (seconds >= -3600)
        & (seconds < 300)
    )
    bins = np.floor((seconds[valid] + 3600) / 5 + 1e-10).astype(int)
    logs = np.log(resistances[valid])
    order = np.argsort(bins, kind="stable")
    bins, logs = bins[order], logs[order]
    unique, starts, counts = np.unique(bins, return_index=True, return_counts=True)
    binned = np.full((TRAIN + H, K), np.nan)
    occupancy = np.zeros(TRAIN + H, dtype=int)
    for index, start, count in zip(unique, starts, counts):
        if 0 <= index < TRAIN + H:
            binned[index] = np.median(logs[start : start + count], axis=0)
            occupancy[index] = count
    return binned, occupancy


def analyze_record(meta, y, occupancy):
    reasons = []
    pairs = np.isfinite(y[: TRAIN - 1]) & np.isfinite(y[1:TRAIN])
    if pairs.sum(axis=0).min() < 300:
        reasons.append("fewer_than_300_adjacent_training_pairs")
    if not np.isfinite(y[TRAIN - 1]).all():
        reasons.append("missing_final_training_bin")
    if not np.isfinite(y[TRAIN:]).all():
        reasons.append("missing_monitoring_bin")
    if meta["category"] != "background" and meta["duration_hours"] < 5 / 60:
        reasons.append("intervention_shorter_than_5min")
    included = dict(
        **meta,
        included=not reasons,
        exclusion=";".join(reasons),
        observed_training_bins=int((occupancy[:TRAIN] > 0).sum()),
        observed_monitoring_bins=int((occupancy[TRAIN:] > 0).sum()),
        min_training_pairs=int(pairs.sum(axis=0).min()),
    )
    if reasons:
        return included, None
    residuals = np.zeros((H, K))
    coef = np.zeros((K, 3))
    diagnostics = []
    for k in range(K):
        good = pairs[:, k]
        previous = y[: TRAIN - 1, k][good]
        target = y[1:TRAIN, k][good]
        design = np.column_stack((np.ones(len(previous)), previous))
        fit, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        center = float(np.median(target - design @ fit))
        coef[k] = [fit[0], fit[1], center]
        residuals[:, k] = y[TRAIN:, k] - fit[0] - fit[1] * y[TRAIN - 1 : -1, k] - center
        pre = residuals[:PRE, k]
        signs = np.sign(pre)
        diagnostics.append(
            dict(
                id=meta["id"],
                date=meta["date"],
                category=meta["category"],
                channel=k + 1,
                ar_intercept=fit[0],
                ar_coefficient=fit[1],
                training_median_residual=center,
                training_design_rank=int(rank),
                pre_positive_fraction=float((signs > 0).mean()),
                pre_negative_fraction=float((signs < 0).mean()),
                pre_zero_fraction=float((signs == 0).mean()),
                pre_mean_sign=float(signs.mean()),
                pre_residual_lag1_acf=acf1(pre),
                pre_sign_lag1_acf=acf1(signs),
            )
        )
    signs = np.sign(residuals)
    evidence = np.zeros((H, K))
    rp, rm = np.zeros(K), np.zeros(K)
    selected = np.zeros((len(METHODS), H, K), dtype=bool)
    ranks = np.arange(1, K + 1)
    threshold = K * H / Q
    for t in range(H):
        rp = (1 + rp) * (1 + 0.5 * signs[t])
        rm = (1 + rm) * (1 - 0.5 * signs[t])
        evidence[t] = 0.5 * (rp + rm)
        ordered = np.sort(evidence[t])[::-1]
        r = int(np.where(ordered * ranks >= threshold, ranks, 0).max())
        if r:
            selected[0, t] = evidence[t] >= threshold / r
        selected[1, t] = evidence[t] >= threshold
        selected[2, t] = evidence[t] >= H / Q
    assert not np.any(selected[1] & ~selected[0])
    outcomes = []
    for j, method in enumerate(METHODS):
        any_alert = selected[j].any(axis=1)
        pre_alert, post_alert = bool(any_alert[:PRE].any()), bool(any_alert[PRE:].any())
        post_times = np.flatnonzero(any_alert[PRE:])
        first_post = 5 * (post_times[0] + 1) if post_times.size else float("nan")
        first = np.flatnonzero(any_alert)
        first_seconds = int(TIMES[first[0]]) if first.size else float("nan")
        outcomes.append(
            dict(
                **meta,
                method=method,
                pre_alert=int(pre_alert),
                any_alert=int(any_alert.any()),
                post_alert=int(post_alert),
                no_pre_alert=int(not pre_alert),
                detected_without_pre_alert=int((not pre_alert) and post_alert),
                first_post_alert_seconds=first_post,
                first_alert_seconds=first_seconds,
                flags_at_5min=int(selected[j, -1].sum()),
            )
        )
    return included, dict(
        residuals=residuals,
        coefficients=coef,
        diagnostics=diagnostics,
        evidence=evidence,
        selected=selected,
        outcomes=outcomes,
    )


def aggregate(outcomes):
    rows = []
    for category in ("all", "stimulus", "wine", "banana", "background"):
        keep = [
            r
            for r in outcomes
            if category == "all"
            or r["category"] == category
            or (category == "stimulus" and r["category"] != "background")
        ]
        for method in METHODS:
            records = [r for r in keep if r["method"] == method]
            if not records:
                continue
            n = len(records)
            row = dict(category=category, method=method, n=n)
            for metric in (
                "pre_alert",
                "any_alert",
                "post_alert",
                "no_pre_alert",
                "detected_without_pre_alert",
            ):
                count = sum(r[metric] for r in records)
                row[metric + "_count"], row[metric + "_fraction"] = count, count / n
            eligible = [r for r in records if r["no_pre_alert"]]
            timely = [r for r in eligible if r["post_alert"]]
            row["eligible_n"] = len(eligible)
            row["timely_n"] = len(timely)
            row["conditional_detection_fraction"] = (
                len(timely) / len(eligible) if eligible else float("nan")
            )
            row["conditional_nondetection_fraction"] = (
                1 - row["conditional_detection_fraction"]
            )
            delays = [r["first_post_alert_seconds"] for r in timely]
            row["conditional_mean_delay_seconds"] = (
                float(np.mean(delays)) if delays else float("nan")
            )
            row["conditional_median_delay_seconds"] = (
                float(np.median(delays)) if delays else float("nan")
            )
            post_delays = [
                r["first_post_alert_seconds"] for r in records if r["post_alert"]
            ]
            row["post_nondetection_fraction"] = 1 - row["post_alert_fraction"]
            row["mean_post_delay_if_detected_seconds"] = (
                float(np.mean(post_delays)) if post_delays else float("nan")
            )
            rows.append(row)
    return rows


def figures(summary, selected, included):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    bg = [
        next(r for r in summary if r["category"] == "background" and r["method"] == m)
        for m in METHODS
    ]
    st = [
        next(r for r in summary if r["category"] == "stimulus" and r["method"] == m)
        for m in METHODS
    ]
    for j in range(len(METHODS)):
        b = bg[j]
        axes[0].bar(j, b["any_alert_fraction"], color=COLORS[j], width=0.58)
        axes[0].text(
            j,
            b["any_alert_fraction"] + 0.025,
            f"{b['any_alert_count']}/{b['n']}",
            ha="center",
        )
        a = st[j]
        value = a["conditional_detection_fraction"]
        if np.isfinite(value):
            axes[1].bar(j, value, color=COLORS[j], width=0.58)
            axes[1].text(
                j, value + 0.025, f"{a['timely_n']}/{a['eligible_n']}", ha="center"
            )
        else:
            axes[1].text(j, 0.06, "No eligible\nrecords", ha="center", fontsize=9)
    labels = ["e-d-BH\n(ours)", "Parallel\nSR", "Per-stream\nSR"]
    for ax in axes:
        ax.set(xticks=range(3), xticklabels=labels, ylim=(0, 1.12), xlim=(-0.5, 2.5))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e7e7e7", lw=0.7)
        ax.set_axisbelow(True)
    axes[0].set(
        title="Background alerts ↓", ylabel="Fraction with any alert in 35 minutes"
    )
    axes[1].set(
        title="Interventions detected within 5 minutes ↑",
        ylabel="Fraction among records with no earlier alert",
    )
    early = ", ".join(f"{a['pre_alert_count']}/{a['n']}" for a in st)
    fig.text(
        0.76,
        0.012,
        "Pre-onset alerts (left to right): " + early,
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=2)
    fig.savefig(OUT / "home_sensor_main.pdf", bbox_inches="tight")
    fig.savefig(OUT / "home_sensor_main.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharey=True)
    for ax, category in zip(axes, ("wine", "banana", "background")):
        ids = [i for i, r in enumerate(included) if r["category"] == category]
        means = selected[ids].mean(axis=(0, 3))
        for j, method in enumerate(METHODS):
            ax.plot(TIMES / 60, means[j], label=method, color=COLORS[j], lw=1.8)
        ax.axvline(0, color="#333333", ls="--", lw=1)
        ax.set(
            title=f"{category.capitalize()} ({len(ids)} records)",
            xlim=(-30, 5),
            ylim=(0, 1.02),
            xlabel="Minutes from reference time",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e7e7e7", lw=0.7)
    axes[0].set_ylabel("Fraction of 8 channels currently flagged")
    axes[2].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "home_sensor_channels.pdf", bbox_inches="tight")
    fig.savefig(OUT / "home_sensor_channels.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata, chosen, raw = load_source()
    all_binned, all_occupancy, inclusions, results = [], [], [], []
    outcomes, diagnostics, included_meta = [], [], []
    for meta in chosen:
        rows = raw[raw[:, 0] == meta["id"]]
        y, occ = bin_record(rows)
        inclusion, result = analyze_record(meta, y, occ)
        all_binned.append(y)
        all_occupancy.append(occ)
        inclusions.append(inclusion)
        if result is not None:
            results.append(result)
            included_meta.append(meta)
            outcomes.extend(result["outcomes"])
            diagnostics.extend(result["diagnostics"])
    write_csv(OUT / "home_sensor_inclusions.csv", inclusions)
    write_csv(OUT / "home_sensor_outcomes.csv", outcomes)
    write_csv(OUT / "home_sensor_diagnostics.csv", diagnostics)
    summary = aggregate(outcomes)
    write_csv(OUT / "home_sensor_summary.csv", summary)
    selected = np.stack([r["selected"] for r in results])
    curves = []
    for category in ("wine", "banana", "background"):
        inds = [i for i, r in enumerate(included_meta) if r["category"] == category]
        average = selected[inds].mean(axis=(0, 3))
        for j, method in enumerate(METHODS):
            for t, seconds in enumerate(TIMES):
                curves.append(
                    dict(
                        category=category,
                        method=method,
                        n=len(inds),
                        seconds=int(seconds),
                        fraction_channels_flagged=float(average[j, t]),
                    )
                )
    write_csv(OUT / "home_sensor_curves.csv", curves)
    np.savez_compressed(
        DATA / "processed_bins.npz",
        ids=np.asarray([r["id"] for r in chosen]),
        log_resistance=np.stack(all_binned),
        occupancy=np.stack(all_occupancy),
        bin_right_seconds=np.arange(-3595, 301, 5),
        included=np.asarray([r["included"] for r in inclusions]),
    )
    np.savez_compressed(
        OUT / "home_sensor_replicates.npz",
        ids=np.asarray([r["id"] for r in included_meta]),
        categories=np.asarray([r["category"] for r in included_meta]),
        dates=np.asarray([r["date"] for r in included_meta]),
        residuals=np.stack([r["residuals"] for r in results]),
        coefficients=np.stack([r["coefficients"] for r in results]),
        evidence=np.stack([r["evidence"] for r in results]),
        selected=selected,
        methods=np.asarray(METHODS),
        bin_right_seconds=TIMES,
    )
    manifest = dict(
        dataset="Gas sensors for home activity monitoring",
        uci_id=362,
        citation="Huerta, F., and Huerta, R. (2016). UCI Machine Learning Repository.",
        doi="10.24432/C5BS4F",
        source_page="https://archive.ics.uci.edu/dataset/362",
        download_url="https://archive.ics.uci.edu/static/public/362/gas%2Bsensors%2Bfor%2Bhome%2Bactivity%2Bmonitoring.zip",
        source_sha256=EXPECTED_SHA,
        source_bytes=(DATA / "source.zip").stat().st_size,
        design_sha256=hashlib.sha256(
            (ROOT / "home_sensor_design.txt").read_bytes()
        ).hexdigest(),
        date_cutoff="2015-08-20",
        selected_records=len(chosen),
        included_records=len(results),
        selected_ids=[r["id"] for r in chosen],
        included_ids=[r["id"] for r in included_meta],
        exclusions=[r for r in inclusions if not r["included"]],
        k=K,
        bin_seconds=5,
        training_bins=TRAIN,
        monitoring_bins=H,
        pre_monitoring_bins=PRE,
        q=Q,
        alpha=Q / H,
        sign_bet=0.5,
        methods=METHODS,
        python=platform.python_version(),
        numpy=np.__version__,
        matplotlib=matplotlib.__version__,
        scope="Model-based descriptive illustration. No empirical sensor-level FDR or guaranteed EOP is claimed.",
    )
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    figures(summary, selected, included_meta)
    print(
        f"Selected{len(chosen)}, included{len(results)}, excluded{len(chosen)-len(results)} records."
    )
    for r in inclusions:
        if not r["included"]:
            print("EXCLUDED", r["id"], r["exclusion"])
    for r in summary:
        if r["category"] in ("stimulus", "background"):
            print(r)
    print(
        "Saved all source provenance, processed observations, diagnostics and outcomes."
    )


if __name__ == "__main__":
    main()
