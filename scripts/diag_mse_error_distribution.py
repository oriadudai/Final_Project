"""Per-window error distribution for the MSE-vs-Huber loss-ablation retrain
(results/comparison_reheartnet_optunalr).

Motivation: while loss_mse_optunalr was training (fold 0), the train/val gap
visibly WIDENED over a short epoch span -- train kept falling (0.6220->0.6129)
while val turned and started rising (0.7862->0.7931). loss_huber_optunalr's
gap stayed flat over the same span (~0.015).

Hypothesis: MSE's gradient is unbounded and proportional to error
(d/dpred = 2*error), so a small tail of hard/atypical training windows
(motion artifacts, noisy beats, etc.) can dominate the gradient and get
memorized -- producing a heavy-tailed per-window error distribution on train
(most windows fit very well, a few stay high) that doesn't transfer to val's
different hard windows. Huber's gradient saturates at +/-delta once
|error| > huber_delta (~1.71), capping any single window's influence and
acting as implicit regularization against this.

For one fold (default: 0) and the same train/val split (seed=42, reproduced
via split_train_val -- identical for reheartnet_original/reheartnet_huber
since both share window_sec=4.0/overlap=0%/bandpass=True), loads each model's
"best" checkpoint and computes per-window MSE (mean squared error per window,
matching nn.MSELoss()'s per-batch value at batch_size=1) on:
  - train_inner (90% of train-subject windows, what the optimizer saw)
  - val_ds      (10% of train-subject windows, held out)
  - test_ds     (held-out subjects' windows, for reference)

Reports mean/median/std/percentiles and the share of total summed loss
contributed by the worst 5%/10% of windows (uniform would be 5%/10%; a much
larger share indicates a heavy tail).

Usage:
    python scripts/diag_mse_error_distribution.py
    python scripts/diag_mse_error_distribution.py --fold 0 --models reheartnet_original,reheartnet_huber
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.config as config
from core.data_loader import build_group_fold, split_train_val
from core.models.baselines import get_model
from compare_reheartnet import MODELS


def per_window_mse(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    losses = []
    with torch.no_grad():
        for ppg, ecg in loader:
            ppg, ecg = ppg.to(device), ecg.to(device)
            pred = model(ppg)
            losses.append(torch.mean((pred - ecg) ** 2).item())
    return np.array(losses)


def describe(losses, label):
    total = losses.sum()
    order = np.argsort(losses)[::-1]
    n5  = max(1, int(round(len(losses) * 0.05)))
    n10 = max(1, int(round(len(losses) * 0.10)))
    top5_share  = losses[order[:n5]].sum()  / total
    top10_share = losses[order[:n10]].sum() / total
    print(f"    {label}: n={len(losses)}  mean={losses.mean():.4f}  median={np.median(losses):.4f}  "
          f"std={losses.std():.4f}  max={losses.max():.4f}")
    pcts = np.percentile(losses, [50, 75, 90, 95, 99])
    print(f"      p50={pcts[0]:.4f}  p75={pcts[1]:.4f}  p90={pcts[2]:.4f}  "
          f"p95={pcts[3]:.4f}  p99={pcts[4]:.4f}")
    print(f"      worst 5% of windows  -> {top5_share:.1%} of total loss  (uniform: 5%)")
    print(f"      worst 10% of windows -> {top10_share:.1%} of total loss  (uniform: 10%)")
    return {
        "n": len(losses), "mean": float(losses.mean()), "median": float(np.median(losses)),
        "std": float(losses.std()), "max": float(losses.max()),
        "p50": float(pcts[0]), "p75": float(pcts[1]), "p90": float(pcts[2]),
        "p95": float(pcts[3]), "p99": float(pcts[4]),
        "top5pct_share": float(top5_share), "top10pct_share": float(top10_share),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default=os.path.join("results", "comparison_reheartnet_optunalr"))
    ap.add_argument("--fold-assignments", type=str, default=None,
                    help="Defaults to <results-dir>/fold_assignments.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--models", type=str, default="reheartnet_original,reheartnet_huber")
    ap.add_argument("--output-dir", type=str, default=os.path.join("results", "diag_mse_error_distribution"))
    args = ap.parse_args()

    device = config.DEVICE
    print(f"Device: {device}")

    fold_assignments_path = args.fold_assignments or os.path.join(args.results_dir, "fold_assignments.json")
    with open(fold_assignments_path) as f:
        assignments = json.load(f)
    fold_key = f"fold_{args.fold:02d}"
    train_subs = assignments[fold_key]["train"]
    test_subs  = assignments[fold_key]["test"]
    print(f"Fold {args.fold}: {len(train_subs)} train subjects, {len(test_subs)} test subjects")

    dataset_cache = {}
    results = {}
    for model_key in [k.strip() for k in args.models.split(",")]:
        model_cfg = MODELS[model_key]
        window_sec     = model_cfg.get("window_sec")
        overlap_frac   = model_cfg.get("overlap_frac", 0.5)
        apply_bandpass = model_cfg.get("apply_bandpass", False)
        hidden_size    = model_cfg.get("hidden_size", config.HIDDEN_SIZE)

        ckpt_path = os.path.join(args.results_dir, model_key, "checkpoints",
                                  f"reheartnet_fold_{args.fold:02d}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"\n{model_key}: checkpoint not found ({ckpt_path}), skipping.")
            continue

        cache_key = (window_sec, overlap_frac, apply_bandpass)
        if cache_key not in dataset_cache:
            train_ds, test_ds = build_group_fold(
                train_subs, test_subs, apply_align=True,
                window_sec=window_sec, overlap_frac=overlap_frac, apply_bandpass=apply_bandpass,
            )
            train_inner, val_ds = split_train_val(train_ds, val_fraction=0.1, seed=42)
            dataset_cache[cache_key] = (train_inner, val_ds, test_ds)
        train_inner, val_ds, test_ds = dataset_cache[cache_key]

        ckpt = torch.load(ckpt_path, map_location=device)
        hidden_size = ckpt.get("hidden_size", hidden_size)
        model = get_model("reheartnet", hidden_size=hidden_size).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        print(f"\n{'-'*60}\n  {model_cfg['label']}  "
              f"(epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss'):.5f})\n{'-'*60}")

        train_losses = per_window_mse(model, train_inner, device)
        val_losses   = per_window_mse(model, val_ds, device)
        test_losses  = per_window_mse(model, test_ds, device)

        results[model_key] = {
            "label": model_cfg["label"], "ckpt_epoch": ckpt.get("epoch"),
            "ckpt_val_loss": ckpt.get("val_loss"),
            "train_inner": describe(train_losses, "train_inner (90% of train-subject windows)"),
            "val_ds":       describe(val_losses,   "val_ds      (10% of train-subject windows)"),
            "test_ds":      describe(test_losses,  "test_ds     (held-out subjects)"),
        }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"fold{args.fold:02d}_error_distribution.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
