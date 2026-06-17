"""Per-class diagnostic error heatmap: model x calibration loss.

Each cell = mean |true_prob - pred_prob| for one pathology superclass,
averaged over all subjects. Rows = (model, calibration) combinations plus
a "no calib" baseline row for each model. Lower = better (green).

Usage:
    python scripts/plot_diag_class_heatmap.py \
        results/diag_calib_loss_ablation_mse_20ep/calib_loss_ablation_summary.json:MSE \
        results/diag_calib_loss_ablation_huber_20ep/calib_loss_ablation_summary.json:Huber \
        results/diag_calib_loss_ablation_composite_20ep/calib_loss_ablation_summary.json:Composite \
        results/diag_calib_loss_ablation_twophase/calib_loss_ablation_summary.json:Two-phase \
        --out results/figures/diag_class_heatmap.pdf
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DEFAULT_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

_MODEL_DISPLAY = {
    "reheartnet_original": "ReHeartNet\n(original)",
    "reheartnet_huber":    "ReHeartNet\n+ Huber",
    "arch_reheartnet":     "ReHeartNet\n+ CLEF (arch)",
}

# preferred display order
_MODEL_ORDER = ["reheartnet_original", "reheartnet_huber", "arch_reheartnet"]


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _per_class_error(per_subject, prob_key):
    """Mean |true_probs_mean - <prob_key>| per class across subjects."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs", nargs="+",
        help="JSON paths, each optionally suffixed with :Label  (e.g. results/...json:MSE). "
             "Label defaults to the parent directory name.",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model keys to include (default: all, in preferred order).",
    )
    parser.add_argument(
        "--out", type=str,
        default=os.path.join("results", "figures", "diag_class_heatmap.pdf"),
    )
    parser.add_argument("--vmax", type=float, default=0.45,
                        help="Colorbar maximum (default 0.45).")
    args = parser.parse_args()

    # Parse path:label pairs
    file_label_pairs = []
    for inp in args.inputs:
        if ":" in inp:
            path, label = inp.rsplit(":", 1)
        else:
            path = inp
            label = os.path.basename(os.path.dirname(inp))
            label = label.replace("diag_calib_loss_ablation_", "").replace("_", " ")
        file_label_pairs.append((path, label))

    all_data = []
    for path, label in file_label_pairs:
        try:
            all_data.append((label, _load_json(path)))
        except FileNotFoundError:
            print(f"WARNING: not found, skipping: {path}")

    if not all_data:
        print("No JSON files loaded.")
        sys.exit(1)

    # Determine model order
    seen_keys = []
    for _, data in all_data:
        for k in data:
            if k not in seen_keys:
                seen_keys.append(k)

    if args.models:
        model_keys = [m.strip() for m in args.models.split(",")]
    else:
        # preferred order first, then any extras
        model_keys = [k for k in _MODEL_ORDER if k in seen_keys]
        model_keys += [k for k in seen_keys if k not in model_keys]

    superclasses = _DEFAULT_SUPERCLASSES

    # Build rows: (display_label, error_array, model_key)
    rows = []
    baseline_seen = set()

    for model_key in model_keys:
        display = _MODEL_DISPLAY.get(model_key, model_key)

        for calib_label, data in all_data:
            if model_key not in data:
                continue
            per_subject = data[model_key].get("per_subject", [])
            if not per_subject:
                continue

            # "no calib" row — add once per model from first occurrence
            if model_key not in baseline_seen:
                err = _per_class_error(per_subject, "baseline_pred_probs_mean")
                if err is not None:
                    rows.append((f"{display}\n(no calib)", err, model_key))
                    baseline_seen.add(model_key)

            # calibrated row
            err = _per_class_error(per_subject, "calibrated_pred_probs_mean")
            if err is not None:
                rows.append((f"{display}\n+ {calib_label}", err, model_key))

    if not rows:
        print("No per-subject diag data found in any JSON.")
        sys.exit(1)

    matrix    = np.array([r[1] for r in rows])   # (n_rows, 5)
    row_labels = [r[0] for r in rows]
    row_groups = [r[2] for r in rows]

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(7, max(3.5, n_rows * 0.62 + 1.8)))

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=args.vmax)

    ax.set_xticks(range(len(superclasses)))
    ax.set_xticklabels(superclasses, fontsize=11, fontweight="bold")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8.5)

    # Annotate cells
    for i in range(n_rows):
        for j in range(len(superclasses)):
            val = matrix[i, j]
            color = "white" if val > args.vmax * 0.65 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=color)

    # Separator lines between model groups
    prev = None
    for i, grp in enumerate(row_groups):
        if prev is not None and grp != prev:
            ax.axhline(i - 0.5, color="white", linewidth=2.5)
        prev = grp

    # Shade "no calib" rows lightly
    for i, lbl in enumerate(row_labels):
        if "no calib" in lbl:
            ax.axhspan(i - 0.5, i + 0.5, color="gray", alpha=0.15, zorder=0)

    cbar = plt.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("Mean |true prob − pred prob|", fontsize=9)

    ax.set_title("Per-class diagnostic error  ·  model × calibration loss",
                 pad=14, fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight", dpi=150)
    print(f"Saved -> {args.out}")
    plt.close()


if __name__ == "__main__":
    main()
