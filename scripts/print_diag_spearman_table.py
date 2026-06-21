"""Print per-class Spearman rank-correlation table (ASCII + LaTeX) from
calibration JSONs produced by diag_calib_loss_ablation.py --diagnostic-classifier.

For each (model, calibration) row and each PTB-XL superclass, computes the
Spearman correlation rho across subjects between the classifier's per-subject
mean probability on the real ECG (true_probs_mean) and on the reconstruction
(baseline_pred_probs_mean for the "no calib" row, calibrated_pred_probs_mean
for the named calibration row). This is the threshold-free complement to
print_diag_class_table.py's mean-absolute-error table: it measures whether
calibration preserves the cross-subject ORDERING of pathology severity,
not the absolute probability values.

Significance uses scipy's asymptotic t-approximation, two-tailed,
unadjusted: *p<0.05, **p<0.01, ***p<0.001. The "Avg" column is the
unweighted mean rho across the 5 classes and is never starred.

Usage:
    python scripts/print_diag_spearman_table.py \
        path/mse.json:MSE path/huber.json:Huber path/composite.json:Composite
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

_MODEL_DISPLAY = {
    "reheartnet_original": "ReHeartNet-MSE",
    "reheartnet_huber":    "ReHeartNet-Huber",
    "arch_reheartnet":     "ReHeartNet-CLEF",
}
_MODEL_ORDER = ["reheartnet_original", "reheartnet_huber", "arch_reheartnet"]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _collect_vectors(per_subject, pred_key):
    true_list, pred_list = [], []
    for entry in per_subject:
        diag = entry.get("diag")
        if not diag:
            continue
        true_p = diag.get("true_probs_mean")
        pred_p = diag.get(pred_key)
        if true_p is None or pred_p is None:
            continue
        true_list.append(true_p)
        pred_list.append(pred_p)
    if not true_list:
        return None, None
    return np.array(true_list), np.array(pred_list)


def _spearman_per_class(true_arr, pred_arr):
    rhos, pvals = [], []
    for c in range(true_arr.shape[1]):
        rho, p = spearmanr(true_arr[:, c], pred_arr[:, c])
        rhos.append(rho)
        pvals.append(p)
    return np.array(rhos), np.array(pvals)


def _stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="JSON:Label pairs")
    args = ap.parse_args()

    file_label_pairs = []
    for inp in args.inputs:
        if ":" in inp:
            path, label = inp.rsplit(":", 1)
        else:
            path = inp
            label = os.path.basename(os.path.dirname(inp))
        file_label_pairs.append((path, label))

    all_data = [(lbl, _load(p)) for p, lbl in file_label_pairs]

    # rows: (model_key, row_label, rhos, pvals, n_subjects)
    rows = []
    baseline_done = set()

    for model_key in _MODEL_ORDER:
        for calib_label, data in all_data:
            if model_key not in data:
                continue
            ps = data[model_key].get("per_subject", [])
            if not ps:
                continue

            if model_key not in baseline_done:
                t, p = _collect_vectors(ps, "baseline_pred_probs_mean")
                if t is not None:
                    rhos, pvals = _spearman_per_class(t, p)
                    rows.append((model_key, "no calib", rhos, pvals, len(t)))
                baseline_done.add(model_key)

            t, p = _collect_vectors(ps, "calibrated_pred_probs_mean")
            if t is not None:
                rhos, pvals = _spearman_per_class(t, p)
                rows.append((model_key, calib_label, rhos, pvals, len(t)))

    if not rows:
        print("No data found.")
        sys.exit(1)

    sc = _SUPERCLASSES
    col_w = 12

    # ── ASCII table ──────────────────────────────────────────────────────────
    header = f"{'Model':<18} {'Calib':<12}" + "".join(f"{s:>{col_w}}" for s in sc) + f"{'Avg':>{col_w}}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    prev_model = None
    for model_key, row_label, rhos, pvals, n in rows:
        mdisplay = _MODEL_DISPLAY.get(model_key, model_key)
        if prev_model != model_key:
            if prev_model is not None:
                print()
            prev_model = model_key
        cells = "".join(f"{rho:+.3f}{_stars(p):<3}".rjust(col_w) for rho, p in zip(rhos, pvals))
        avg = rhos.mean()
        print(f"{mdisplay:<18} {row_label:<12}{cells}{avg:>{col_w}.3f}  (n={n})")

    print(sep)

    # ── LaTeX table ──────────────────────────────────────────────────────────
    print("\n% LaTeX table -- Spearman rank correlation (true vs. reconstructed per-subject class probability)")
    print(r"\begin{tabular}{ll" + "c" * len(sc) + "r}")
    print(r"\toprule")
    print("Model & Calib & " + " & ".join(sc) + r" & Avg \\")
    print(r"\midrule")

    prev_model = None
    for model_key, row_label, rhos, pvals, n in rows:
        mdisplay = _MODEL_DISPLAY.get(model_key, model_key)
        if prev_model != model_key:
            if prev_model is not None:
                print(r"\addlinespace")
            n_rows = len([r for r in rows if r[0] == model_key])
            print(f"\\multirow{{{n_rows}}}{{*}}{{{mdisplay}}}", end="")
            prev_model = model_key
        else:
            print("", end="")
        cells = " & ".join(f"${rho:+.3f}^{{{_stars(p)}}}$" if _stars(p) else f"${rho:+.3f}$"
                            for rho, p in zip(rhos, pvals))
        avg = rhos.mean()
        print(f" & {row_label} & {cells} & ${avg:+.3f}$ \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
