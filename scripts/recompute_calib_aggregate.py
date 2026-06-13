"""Recompute the "aggregate" block of a diag_calib_loss_ablation.py summary
JSON as mean +/- 95% CI over the saved "per_subject" entries, without
re-running the (expensive) per-subject calibration fine-tuning loop.

Use this after diag_subject_calibration.aggregate_with_ci() changes, to
refresh an existing calib_loss_ablation_summary.json in place.

Usage:
    python scripts/recompute_calib_aggregate.py results/diag_calib_loss_ablation_mse_composite_full_v2/calib_loss_ablation_summary.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diag_subject_calibration import aggregate_with_ci

AGG_KEYS = ("rmse", "prd", "pearson_r", "emd", "ks_stat", "beat_timing_mae")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary_json")
    ap.add_argument("--output", type=str, default=None,
                    help="Defaults to overwriting the input file in place.")
    args = ap.parse_args()

    with open(args.summary_json) as f:
        results = json.load(f)

    for model_key, entry in results.items():
        per_subject = entry["per_subject"]
        entry["aggregate"] = aggregate_with_ci(per_subject, AGG_KEYS)

        agg = entry["aggregate"]
        print(f"=== {model_key}: {entry['n_subjects']} subjects across {entry['n_folds']} fold(s) ===")
        for split, m in (("baseline  ", agg["baseline"]), ("calibrated", agg["calibrated"])):
            print(f"  {split}: RMSE={m['rmse']['mean']:.4f}+/-{m['rmse']['ci95_margin']:.4f}  "
                  f"PRD={m['prd']['mean']:6.2f}+/-{m['prd']['ci95_margin']:.2f}%  "
                  f"r={m['pearson_r']['mean']:+.4f}+/-{m['pearson_r']['ci95_margin']:.4f}  "
                  f"EMD={m['emd']['mean']:.4f}+/-{m['emd']['ci95_margin']:.4f}  "
                  f"KS={m['ks_stat']['mean']:.3f}+/-{m['ks_stat']['ci95_margin']:.3f}  "
                  f"beat-MAE={m['beat_timing_mae']['mean']:.4f}+/-{m['beat_timing_mae']['ci95_margin']:.4f}s")

    out_path = args.output or args.summary_json
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
