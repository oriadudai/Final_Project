"""Print per-class diagnostic error table (ASCII + LaTeX) from calibration JSONs.

Usage:
    python scripts/print_diag_class_table.py \
        path/mse.json:MSE path/huber.json:Huber path/composite.json:Composite \
        [--delta]   # show improvement over no-calib baseline instead of absolute
"""

import argparse
import json
import os
import sys

import numpy as np

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


def _per_class_error(per_subject, prob_key):
    errs = []
    for entry in per_subject:
        diag = entry.get("diag")
        if not diag:
            continue
        true_p = diag.get("true_probs_mean")
        pred_p = diag.get(prob_key)
        if true_p is None or pred_p is None:
            continue
        errs.append(np.abs(np.array(true_p) - np.array(pred_p)))
    return np.mean(errs, axis=0) if errs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="JSON:Label pairs")
    ap.add_argument("--delta", action="store_true",
                    help="Show delta (baseline − calibrated); positive = improvement")
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

    # Collect rows: (model_key, calib_label, baseline_err, calib_err)
    rows = []
    baseline_cache = {}  # model_key -> baseline error array

    for model_key in _MODEL_ORDER:
        for calib_label, data in all_data:
            if model_key not in data:
                continue
            ps = data[model_key].get("per_subject", [])
            if not ps:
                continue

            b_err = _per_class_error(ps, "baseline_pred_probs_mean")
            c_err = _per_class_error(ps, "calibrated_pred_probs_mean")

            if b_err is not None and model_key not in baseline_cache:
                baseline_cache[model_key] = b_err

            if c_err is not None:
                rows.append((model_key, calib_label, b_err, c_err))

    if not rows:
        print("No data found.")
        sys.exit(1)

    sc = _SUPERCLASSES
    col_w = 8

    # ── ASCII table ──────────────────────────────────────────────────────────
    header = f"{'Model':<18} {'Calib':<12}" + "".join(f"{s:>{col_w}}" for s in sc) + f"{'Avg':>{col_w}}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    prev_model = None
    for model_key, calib_label, b_err, c_err in rows:
        mdisplay = _MODEL_DISPLAY.get(model_key, model_key)

        if prev_model != model_key:
            if prev_model is not None:
                print()
            # print baseline row
            if b_err is not None:
                vals = b_err
                avg  = vals.mean()
                row  = f"{'  (no calib)':<18} {'':<12}" + "".join(f"{v:>{col_w}.3f}" for v in vals) + f"{avg:>{col_w}.3f}"
                print(f"{mdisplay:<18} {row[18:]}")
            prev_model = model_key

        if args.delta:
            # positive = improvement
            b = baseline_cache.get(model_key, b_err)
            vals = b - c_err if b is not None else c_err
        else:
            vals = c_err

        avg = vals.mean()
        row = f"{'':>18} {calib_label:<12}" + "".join(f"{v:>{col_w}.3f}" for v in vals) + f"{avg:>{col_w}.3f}"
        print(f"{mdisplay:<18}{row[18:]}")

    print(sep)

    # ── LaTeX table ──────────────────────────────────────────────────────────
    mode = "delta (positive = improvement)" if args.delta else "absolute error"
    print(f"\n% LaTeX table — per-class mean |true - pred| ({mode})")
    print(r"\begin{tabular}{ll" + "r" * len(sc) + "r}")
    print(r"\toprule")
    print("Model & Calib & " + " & ".join(sc) + r" & Avg \\")
    print(r"\midrule")

    prev_model = None
    for model_key, calib_label, b_err, c_err in rows:
        mdisplay = _MODEL_DISPLAY.get(model_key, model_key)

        if prev_model != model_key:
            if prev_model is not None:
                print(r"\addlinespace")
            # baseline
            if b_err is not None:
                vals = b_err
                avg  = vals.mean()
                cells = " & ".join(f"{v:.3f}" for v in vals)
                print(f"\\multirow{{{len([r for r in rows if r[0]==model_key])+1}}}{{*}}{{{mdisplay}}} & no calib & {cells} & {avg:.3f} \\\\")
            prev_model = model_key
            model_printed = True

        if args.delta:
            b = baseline_cache.get(model_key, b_err)
            vals = b - c_err if b is not None else c_err
        else:
            vals = c_err

        avg = vals.mean()
        cells = " & ".join(f"{v:.3f}" for v in vals)
        print(f" & {calib_label} & {cells} & {avg:.3f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
