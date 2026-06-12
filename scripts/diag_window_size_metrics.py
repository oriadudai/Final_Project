"""Isolate whether reheartnet_huber/reheartnet_original's worse EMD/KS (vs.
reheartnet_clef) is a loss-function effect or an evaluation-window-size
confound.

In `compare_reheartnet.py`, reheartnet_original/reheartnet_huber are trained
AND evaluated on 4 s/non-overlapping/bandpassed windows (Lee et al. protocol),
while reheartnet_clef is trained AND evaluated on 10 s/50%-overlap/no-bandpass
windows (CLEF encoder requirement). Their EMD/KS results (computed on
RR-interval distributions extracted via R-peak detection) differ a lot:

    original : EMD=0.174  KS=0.530
    huber    : EMD=0.119  KS=0.373
    clef     : EMD=0.046  KS=0.112

Window size could plausibly drive much of this gap independent of the loss:
shorter, non-overlapping 4 s windows contain fewer heartbeats, so RR-interval
distributions extracted per-window are noisier/sparser, inflating EMD/KS
regardless of how good the waveform reconstruction is.

This script takes the SAME trained checkpoints (original/huber, unchanged)
and re-evaluates each on a SECOND test set built with clef's window settings
(10 s, 50% overlap, no bandpass), via `build_test_dataset`. Comparing each
checkpoint's "own-window" metrics to its "clef-matched-window" metrics
isolates the window-size effect with everything else (loss function,
training data, weights) held fixed.

Usage:
    python scripts/diag_window_size_metrics.py --clef-path models/clef/clef_small.ckpt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.config as config
from core.data_loader import build_test_dataset
from core.evaluate import evaluate_fold
from core.losses.composite_loss import load_clef_encoder
from core.models.baselines import get_model
from core.models.ptbxl_classifier import get_ptbxl_classifier
from compare_reheartnet import MODELS

AGG_METRICS = ["rmse", "prd", "pearson_r", "bce", "emd", "ks_stat", "beat_timing_mae"]


def _mean_ci(values):
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return float("nan"), 0.0
    mean = float(np.mean(arr))
    margin = 1.96 * float(np.std(arr, ddof=1)) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="reheartnet_original,reheartnet_huber",
                    help="Comma-separated keys from compare_reheartnet.MODELS to re-evaluate.")
    ap.add_argument("--results-dir", type=str, default=os.path.join("results", "comparison_reheartnet"),
                    help="Directory containing {model_key}/checkpoints/ and fold_assignments.json.")
    ap.add_argument("--fold-assignments", type=str, default=None,
                    help="Defaults to <results-dir>/fold_assignments.json")
    ap.add_argument("--folds", type=str, default=None,
                    help="Comma-separated fold indices to run (default: all folds in fold_assignments)")
    ap.add_argument("--clef-path", type=str, default=None)
    ap.add_argument("--clef-dir", type=str, default=config.CLEF_CHECKPOINT_DIR)
    ap.add_argument("--clef-size", type=str, default="auto", choices=["auto", "small", "medium", "large"])
    ap.add_argument("--classifier-path", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--output-dir", type=str, default=os.path.join("results", "diag_window_size_metrics"))
    args = ap.parse_args()

    device = config.DEVICE
    print(f"Device: {device}")

    if args.clef_size == "auto":
        args.clef_size = "medium" if device.type == "cuda" else "small"
        print(f"CLEF size auto-selected: {args.clef_size}")
    if args.clef_path is None:
        args.clef_path = os.path.join(args.clef_dir, f"clef_{args.clef_size}.ckpt")
        print(f"CLEF path auto-set: {args.clef_path}")

    print(f"Loading CLEF encoder ({args.clef_size}) ...")
    clef_encoder = load_clef_encoder(args.clef_path, args.clef_size, device)
    ptbxl_clf = get_ptbxl_classifier(args.classifier_path).to(device)

    fold_assignments_path = args.fold_assignments or os.path.join(args.results_dir, "fold_assignments.json")
    with open(fold_assignments_path) as f:
        assignments = json.load(f)
    fold_keys = sorted(assignments.keys())
    if args.folds:
        wanted = {int(x) for x in args.folds.split(",")}
        fold_keys = [k for k in fold_keys if int(k.split("_")[1]) in wanted]
    print(f"Folds: {fold_keys}")

    model_keys = [k.strip() for k in args.models.split(",")]

    results = {}
    for model_key in model_keys:
        if model_key not in MODELS:
            print(f"\nSkipping unknown model key: {model_key} (not in compare_reheartnet.MODELS)")
            continue
        model_cfg = MODELS[model_key]
        own_window_sec = model_cfg.get("window_sec")
        own_overlap_frac = model_cfg.get("overlap_frac", 0.5)
        own_bandpass = model_cfg.get("apply_bandpass", False)

        if own_window_sec is None:
            print(f"\nSkipping {model_key}: already trained/evaluated on 10s windows, nothing to isolate.")
            continue

        print(f"\n{'-'*60}")
        print(f"  Model: {model_cfg['label']}")
        print(f"  own window:          {own_window_sec:g}s  overlap={own_overlap_frac:.0%}  bandpass={own_bandpass}")
        print(f"  clef-matched window: 10s  overlap=50%  bandpass=False")
        print(f"{'-'*60}")

        per_fold = []
        for fold_key in fold_keys:
            fold_idx = int(fold_key.split("_")[1])
            test_subs = assignments[fold_key]["test"]

            ckpt_path = os.path.join(args.results_dir, model_key, "checkpoints",
                                      f"reheartnet_fold_{fold_idx:02d}_best.pt")
            if not os.path.exists(ckpt_path):
                print(f"  fold {fold_idx:02d}: checkpoint not found ({ckpt_path}), skipping.")
                continue

            ckpt = torch.load(ckpt_path, map_location=device)
            hidden_size = ckpt.get("hidden_size", model_cfg.get("hidden_size", config.HIDDEN_SIZE))
            model = get_model("reheartnet", hidden_size=hidden_size).to(device)
            model.load_state_dict(ckpt["model_state_dict"])

            own_ds = build_test_dataset(test_subs, window_sec=own_window_sec,
                                         overlap_frac=own_overlap_frac, apply_bandpass=own_bandpass)
            own_loader = DataLoader(own_ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=0, pin_memory=True)
            own_metrics = evaluate_fold(model, own_loader, ptbxl_clf, device, clef_encoder=None)

            clef_ds = build_test_dataset(test_subs, window_sec=None, overlap_frac=0.5, apply_bandpass=False)
            clef_loader = DataLoader(clef_ds, batch_size=args.batch_size, shuffle=False,
                                      num_workers=0, pin_memory=True)
            clef_metrics = evaluate_fold(model, clef_loader, ptbxl_clf, device, clef_encoder=clef_encoder)

            print(f"  fold {fold_idx:02d} ({len(test_subs)} subjects, ckpt epoch={ckpt.get('epoch')}):")
            print(f"    own ({own_window_sec:g}s/{own_overlap_frac:.0%}/bp={own_bandpass}): "
                  f"PRD={own_metrics['prd']:6.2f}%  r={own_metrics['pearson_r']:+.3f}  "
                  f"EMD={own_metrics['emd']:.4f}  KS={own_metrics['ks_stat']:.3f}  "
                  f"beat-MAE={own_metrics['beat_timing_mae']:.4f}s")
            print(f"    10s/50%/bp=False           : "
                  f"PRD={clef_metrics['prd']:6.2f}%  r={clef_metrics['pearson_r']:+.3f}  "
                  f"EMD={clef_metrics['emd']:.4f}  KS={clef_metrics['ks_stat']:.3f}  "
                  f"beat-MAE={clef_metrics['beat_timing_mae']:.4f}s")

            per_fold.append({
                "fold": fold_idx, "subjects": test_subs,
                "own": own_metrics, "clef_matched": clef_metrics,
            })

        if not per_fold:
            print(f"  No checkpoints found for {model_key}, skipping aggregate.")
            continue

        aggregate = {}
        for window_key in ("own", "clef_matched"):
            aggregate[window_key] = {}
            for m in AGG_METRICS:
                mean, margin = _mean_ci([f[window_key][m] for f in per_fold])
                aggregate[window_key][m] = {"mean": mean, "ci95_margin": margin}

        print(f"\n  === {model_key} aggregate (mean over {len(per_fold)} fold(s)) ===")
        for window_key, label in (("own", f"own ({own_window_sec:g}s)"), ("clef_matched", "10s/50%/bp=False")):
            m = aggregate[window_key]
            print(f"    {label:18s}: PRD={m['prd']['mean']:6.2f}%  r={m['pearson_r']['mean']:+.3f}  "
                  f"EMD={m['emd']['mean']:.4f}  KS={m['ks_stat']['mean']:.3f}  "
                  f"beat-MAE={m['beat_timing_mae']['mean']:.4f}s")

        results[model_key] = {
            "label": model_cfg["label"],
            "own_window": {"window_sec": own_window_sec, "overlap_frac": own_overlap_frac,
                           "apply_bandpass": own_bandpass},
            "per_fold": per_fold,
            "aggregate": aggregate,
        }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "window_size_isolation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
