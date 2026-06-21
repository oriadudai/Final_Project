"""Build the distortion-perception plane figure (PRD vs EMD by default) from
one or more diag_calib_loss_ablation.py summary JSONs -- one JSON per
calibration objective (--run Label=path), each containing one or more model
keys with "aggregate"."baseline"/"calibrated" metrics.

Locked design (see memory: project_distortion_perception_plot_design):
PRD-vs-EMD plane only. diag_kl/flip_rate are deliberately NOT plotted here --
they stay in the per-class diagnostic tables (tab:diag_class_delta /
tab:diag_spearman), where their CD-specific finding survives; collapsing
them to a single scalar per model x condition for this plot would throw
that away.

Usage:
    python scripts/plot_distortion_perception_plane.py \\
        --run MSE=results/diag_calib_loss_ablation_arch_mse_n41/calib_loss_ablation_summary.json \\
        --run Huber=results/diag_calib_loss_ablation_arch_huber_n41/calib_loss_ablation_summary.json \\
        --run Composite=results/diag_calib_loss_ablation_arch_composite_n41/calib_loss_ablation_summary.json \\
        --model reheartnet_original="ReHeartNet (original)" \\
        --model reheartnet_huber="ReHeartNet + Huber" \\
        --model arch_reheartnet="ReHeartNet + CLEF"
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.visualization.plots import plot_distortion_perception_plane


def _parse_kv_list(items):
    out = {}
    for item in items:
        key, value = item.split("=", 1)
        out[key] = value
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="CalibLabel=path/to/calib_loss_ablation_summary.json "
                         "(repeatable, one per calibration objective).")
    ap.add_argument("--model", action="append", required=True,
                    help="model_key=Display Label (repeatable, one per model variant to plot).")
    ap.add_argument("--x-metric", type=str, default="prd")
    ap.add_argument("--y-metric", type=str, default="emd")
    ap.add_argument("--save-path", type=str, default=None)
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()

    runs = _parse_kv_list(args.run)
    models_wanted = _parse_kv_list(args.model)

    summaries = {}
    for calib_label, path in runs.items():
        with open(path) as f:
            summaries[calib_label] = json.load(f)

    def _point(agg_branch, metric):
        return {
            metric: agg_branch[metric]["mean"],
            f"{metric}_ci": agg_branch[metric]["ci95_margin"],
        }

    models = []
    for model_key, display_label in models_wanted.items():
        baseline = None
        calibrations = []
        for calib_label, summary in summaries.items():
            if model_key not in summary:
                print(f"  [skip] {model_key} not found in run '{calib_label}' ({runs[calib_label]})")
                continue
            agg = summary[model_key]["aggregate"]
            if baseline is None:
                baseline = {**_point(agg["baseline"], args.x_metric),
                            **_point(agg["baseline"], args.y_metric)}
            calibrations.append({
                "label": calib_label,
                **_point(agg["calibrated"], args.x_metric),
                **_point(agg["calibrated"], args.y_metric),
            })
        if baseline is None:
            print(f"  [skip] {model_key}: not found in any run, skipping entirely.")
            continue
        models.append({"label": display_label, "baseline": baseline, "calibrations": calibrations})

    if not models:
        raise SystemExit("No models found in any provided run -- nothing to plot.")

    plot_distortion_perception_plane(
        models, x_metric=args.x_metric, y_metric=args.y_metric,
        save_path=args.save_path, title=args.title,
    )
    out_path = args.save_path or os.path.join("results", "figures", "distortion_perception_plane.png")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
