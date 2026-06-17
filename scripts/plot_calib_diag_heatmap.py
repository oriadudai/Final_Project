"""Heatmap visualisation of per-subject diagnostic-classifier probabilities.

Reads one or more calib_loss_ablation_summary.json files produced by
diag_calib_loss_ablation.py --diagnostic-classifier <path>.

For each (model, condition) pair the script plots:
  • Heatmap A: |true_probs - calibrated_pred_probs| per subject × superclass
  • Heatmap B: binary top-1 flip mask (1 = argmax changed after calibration)
  • Heatmap C: same as A but for baseline (pre-calibration) predictions
  • Bar chart: mean |delta| per superclass, baseline vs calibrated

Usage:
    python scripts/plot_calib_diag_heatmap.py \\
        results/diag_calib_two_phase/calib_loss_ablation_summary.json \\
        --out-dir results/diag_calib_heatmaps

    # Compare multiple runs side-by-side:
    python scripts/plot_calib_diag_heatmap.py \\
        results/diag_calib_loss_ablation_mse/calib_loss_ablation_summary.json \\
        results/diag_calib_two_phase/calib_loss_ablation_summary.json \\
        --labels MSE two-phase \\
        --out-dir results/diag_calib_heatmaps
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


_DEFAULT_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def _load_summary(path: str):
    with open(path) as f:
        return json.load(f)


def _extract_diag_arrays(per_subject, n_classes):
    """Return (subjects, true_probs, bl_probs, cal_probs) numpy arrays.

    Subjects without "diag" keys are silently skipped.
    """
    subj_ids, true_list, bl_list, cal_list = [], [], [], []
    for s in per_subject:
        d = s.get("diag")
        if d is None:
            continue
        subj_ids.append(s["subject"])
        true_list.append(d["true_probs_mean"])
        bl_list.append(d["baseline_pred_probs_mean"])
        cal_list.append(d["calibrated_pred_probs_mean"])
    if not subj_ids:
        return None, None, None, None
    return (
        subj_ids,
        np.array(true_list, dtype=np.float32),
        np.array(bl_list,  dtype=np.float32),
        np.array(cal_list, dtype=np.float32),
    )


def _plot_model(model_key, model_data, run_label, out_dir, superclasses):
    per_subject = model_data.get("per_subject", [])
    n_classes = len(superclasses)
    subj_ids, true_arr, bl_arr, cal_arr = _extract_diag_arrays(per_subject, n_classes)

    if subj_ids is None:
        print(f"  [{model_key}] no diag data — skipping")
        return

    n_subj = len(subj_ids)
    delta_cal = np.abs(true_arr - cal_arr)   # (N, C)
    delta_bl  = np.abs(true_arr - bl_arr)    # (N, C)

    # Binary flip: did the predicted top-1 class match the true top-1?
    flip_bl  = (np.argmax(true_arr, axis=1) != np.argmax(bl_arr,  axis=1)).astype(float)
    flip_cal = (np.argmax(true_arr, axis=1) != np.argmax(cal_arr, axis=1)).astype(float)

    fig, axes = plt.subplots(2, 3, figsize=(16, max(6, n_subj * 0.22 + 2)))
    tag = f"{run_label} / {model_key}" if run_label else model_key
    fig.suptitle(tag, fontsize=11, fontweight="bold")

    def _heatmap(ax, mat, title, vmin=0, vmax=1, cmap="hot_r", fmt=".2f"):
        im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks(range(len(superclasses)))
        ax.set_xticklabels(superclasses, fontsize=8)
        ax.set_yticks(range(n_subj))
        ax.set_yticklabels(subj_ids, fontsize=6)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        for i in range(n_subj):
            for j in range(len(superclasses)):
                val = mat[i, j]
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=5, color="white" if val > 0.5 * vmax else "black")

    _heatmap(axes[0, 0], delta_bl,  "|true − pred| (baseline)")
    _heatmap(axes[0, 1], delta_cal, "|true − pred| (calibrated)")

    # Improvement map: positive = calibration helped
    improvement = delta_bl - delta_cal
    vext = float(np.abs(improvement).max()) or 0.1
    _heatmap(axes[0, 2], improvement, "Improvement (bl − cal |Δ|)",
             vmin=-vext, vmax=vext, cmap="RdYlGn")

    # Flip masks (column vectors, repeated for visibility)
    flip_bl_mat  = flip_bl[:, None].repeat(n_classes, axis=1)
    flip_cal_mat = flip_cal[:, None].repeat(n_classes, axis=1)
    _heatmap(axes[1, 0], flip_bl_mat,  "Top-1 flip (baseline)",  vmin=0, vmax=1, cmap="Reds", fmt=".0f")
    _heatmap(axes[1, 1], flip_cal_mat, "Top-1 flip (calibrated)", vmin=0, vmax=1, cmap="Reds", fmt=".0f")

    # Mean |delta| bar chart per class
    ax_bar = axes[1, 2]
    x = np.arange(n_classes)
    w = 0.35
    ax_bar.bar(x - w/2, delta_bl.mean(axis=0),  w, label="baseline",   color="steelblue", alpha=0.8)
    ax_bar.bar(x + w/2, delta_cal.mean(axis=0), w, label="calibrated", color="tomato",    alpha=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(superclasses, fontsize=8)
    ax_bar.set_ylabel("mean |true − pred|")
    ax_bar.set_title("Mean |Δ| per class", fontsize=9)
    ax_bar.legend(fontsize=7)
    ax_bar.set_ylim(0, 1)

    plt.tight_layout()

    safe_tag = f"{run_label}_{model_key}".replace(" ", "_").replace("/", "-") if run_label else model_key
    out_path = os.path.join(out_dir, f"diag_heatmap_{safe_tag}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    out_png = out_path.replace(".pdf", ".png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{model_key}] saved {out_path}")

    # Summary stats
    pct_bl  = 100.0 * flip_bl.mean()
    pct_cal = 100.0 * flip_cal.mean()
    print(f"    top-1 flip: baseline={pct_bl:.1f}%  calibrated={pct_cal:.1f}%")
    print(f"    mean |Δ|:   baseline={delta_bl.mean():.4f}  calibrated={delta_cal.mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+",
                    help="One or more calib_loss_ablation_summary.json paths.")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Short label for each JSON (same order). "
                         "Defaults to the parent directory name of each JSON.")
    ap.add_argument("--out-dir", type=str, default=os.path.join("results", "diag_calib_heatmaps"))
    ap.add_argument("--models", type=str, default=None,
                    help="Comma-separated model keys to plot (default: all in the JSON).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    labels = args.labels or [os.path.basename(os.path.dirname(p)) for p in args.jsons]
    if len(labels) != len(args.jsons):
        ap.error(f"--labels must have the same length as positional jsons "
                 f"({len(labels)} vs {len(args.jsons)})")

    model_filter = set(args.models.split(",")) if args.models else None

    for json_path, run_label in zip(args.jsons, labels):
        print(f"\n{'='*60}\n{run_label}  ({json_path})\n{'='*60}")
        summary = _load_summary(json_path)

        for model_key, model_data in summary.items():
            if model_filter and model_key not in model_filter:
                continue
            per_subject = model_data.get("per_subject", [])
            superclasses = _DEFAULT_SUPERCLASSES
            for s in per_subject:
                if s.get("diag", {}).get("superclasses"):
                    superclasses = s["diag"]["superclasses"]
                    break

            _plot_model(model_key, model_data, run_label, args.out_dir, superclasses)

    print(f"\nAll heatmaps saved to {args.out_dir}")


if __name__ == "__main__":
    main()
