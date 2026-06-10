"""Diagnostic: shift-corrected clinical metrics for a saved ReHeartNet checkpoint.

Tests whether the model has learned the correct ECG rhythm/shape but is
phase-shifted relative to ground truth at test time -- a predicted
consequence of build_group_fold's train-only phase alignment
(core/data_loader.py:69-76, src/preprocessing.py:51-65 phase_align_ppg).

For each test window, brute-force searches shifts of `pred` in
[-max_lag, +max_lag] samples, picks the shift that minimises
sum((true - roll(pred, k))^2), and reports the FULL clinical metric set
(RMSE, PRD, Pearson r, EMD, KS, beat-timing MAE, and -- for the CLEF/10s
model only -- BCE) before vs after shift-correction, plus the distribution
of chosen shifts.

BCE is skipped for the 4 s-window models (original/huber): CLEFClassifier
resamples its input to a fixed 5000 samples assuming a 10 s/500 Hz signal,
so it can't be applied correctly to 4 s/500-sample windows without building
a separately-windowed 10 s dataset (whose window boundaries -- and thus
per-window shift lags -- wouldn't match the native 4 s windows used here).

Usage:
    python scripts/diag_phase_shift_metrics.py \
        --checkpoint results/comparison_reheartnet/reheartnet_clef/checkpoints/reheartnet_fold_00_best.pt \
        --model-key reheartnet_clef --fold 0
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.config as config
from core.data_loader import build_test_dataset, get_cv_splits
from core.models.baselines import get_model
from core.metrics.clinical_metrics import (
    compute_rmse,
    compute_prd,
    compute_emd,
    compute_ks,
    compute_beat_timing_mae,
    compute_bce,
)
from core.models.ptbxl_classifier import build_classifier
from core.losses.composite_loss import load_clef_encoder
from src.preprocessing import get_all_record_names

# Must match the preprocessing settings for each variant in
# scripts/compare_reheartnet.py's MODELS dict.
PREPROC = {
    "reheartnet_original": dict(window_sec=4.0,  overlap_frac=0.0, apply_bandpass=True),
    "reheartnet_huber":    dict(window_sec=4.0,  overlap_frac=0.0, apply_bandpass=True),
    "reheartnet_clef":     dict(window_sec=None, overlap_frac=0.5, apply_bandpass=False),
}


def best_shift(pred: np.ndarray, true: np.ndarray, max_lag: int) -> tuple:
    """Return (shifted_pred, lag) minimising sum((true - roll(pred, lag))^2)."""
    best_k, best_err = 0, float(np.sum((true - pred) ** 2))
    for k in range(-max_lag, max_lag + 1):
        if k == 0:
            continue
        err = float(np.sum((true - np.roll(pred, k)) ** 2))
        if err < best_err:
            best_err, best_k = err, k
    return np.roll(pred, best_k), best_k


def pearson_r(true_arr: np.ndarray, pred_arr: np.ndarray) -> float:
    vals = []
    for t, p in zip(true_arr, pred_arr):
        if np.std(t) > 1e-8 and np.std(p) > 1e-8:
            vals.append(float(np.corrcoef(t, p)[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def compute_all_metrics(true_arr, pred_arr, fs, classifier=None, device=None) -> dict:
    metrics = {
        "rmse":            compute_rmse(true_arr, pred_arr),
        "prd":             compute_prd(true_arr, pred_arr),
        "pearson_r":       pearson_r(true_arr, pred_arr),
        "emd":             compute_emd(true_arr, pred_arr, fs=fs),
    }
    ks_stat, ks_pval = compute_ks(true_arr, pred_arr, fs=fs)
    metrics["ks_stat"]   = ks_stat
    metrics["ks_pvalue"] = ks_pval
    metrics["beat_timing_mae"] = compute_beat_timing_mae(true_arr, pred_arr, fs=fs)
    if classifier is not None:
        metrics["bce"] = compute_bce(true_arr, pred_arr, classifier, device)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-key", required=True, choices=list(PREPROC))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--max-lag", type=int, default=125)
    ap.add_argument("--clef-path", type=str, default=None,
                     help="Path to CLEF .ckpt for BCE (CLEF model only). "
                          "If omitted, auto-constructs from --clef-dir/--clef-size.")
    ap.add_argument("--clef-size", type=str, default="auto",
                     choices=["auto", "small", "medium", "large"])
    ap.add_argument("--clef-dir", type=str, default=config.CLEF_CHECKPOINT_DIR)
    args = ap.parse_args()

    device = config.DEVICE
    ckpt = torch.load(args.checkpoint, map_location=device)
    hidden_size = ckpt["hidden_size"]
    print(f"hidden_size from checkpoint: {hidden_size}")
    print(f"checkpoint fold: {ckpt['fold']}  epoch: {ckpt['epoch']}  val_loss: {ckpt['val_loss']:.5f}")

    model = get_model("reheartnet", hidden_size=hidden_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_subjects = get_all_record_names()
    splits = get_cv_splits(all_subjects, n_splits=8,
                            save_path=os.path.join("results", "_diag_fold_assignments.json"))
    _, test_subs = splits[args.fold]
    print(f"fold {args.fold} test subjects: {test_subs}")

    pp = PREPROC[args.model_key]
    test_ds = build_test_dataset(test_subs, **pp)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    all_true, all_pred = [], []
    with torch.no_grad():
        for ppg, ecg in loader:
            pred = model(ppg.to(device))
            all_true.append(ecg.squeeze(-1).numpy())
            all_pred.append(pred.squeeze(-1).cpu().numpy())
    true_arr = np.concatenate(all_true, axis=0)
    pred_arr = np.concatenate(all_pred, axis=0)
    print(f"test windows: {true_arr.shape[0]}  window length: {true_arr.shape[1]}")

    # BCE only meaningful for native 10 s windows (CLEFClassifier resamples to a
    # fixed 5000 samples assuming 10 s/500 Hz input).
    classifier = None
    if args.model_key == "reheartnet_clef":
        if args.clef_size == "auto":
            args.clef_size = "medium" if device.type == "cuda" else "small"
        if args.clef_path is None:
            args.clef_path = os.path.join(args.clef_dir, f"clef_{args.clef_size}.ckpt")
        print(f"Loading CLEF encoder ({args.clef_size}) from {args.clef_path} ...")
        clef_encoder = load_clef_encoder(args.clef_path, args.clef_size, device)
        classifier = build_classifier(clef_encoder).to(device)
    else:
        print("BCE skipped (4s-window model -- see module docstring).")

    # Shift-corrected predictions
    shifted = np.empty_like(pred_arr)
    lags = np.empty(len(pred_arr), dtype=int)
    for i in range(len(pred_arr)):
        shifted[i], lags[i] = best_shift(pred_arr[i], true_arr[i], args.max_lag)

    metrics_raw   = compute_all_metrics(true_arr, pred_arr, config.FS, classifier, device)
    metrics_shift = compute_all_metrics(true_arr, shifted,  config.FS, classifier, device)

    print(f"\n=== {args.model_key}  fold {args.fold} ===")
    print(f"{'metric':<16}{'raw':>12}{'shift-corrected':>18}")
    for key in ["rmse", "prd", "pearson_r", "emd", "ks_stat", "ks_pvalue", "beat_timing_mae", "bce"]:
        if key not in metrics_raw:
            continue
        print(f"{key:<16}{metrics_raw[key]:>12.4f}{metrics_shift[key]:>18.4f}")

    print(f"\nLag distribution (samples, +/-{args.max_lag}):")
    print(f"  mean={lags.mean():.1f}  std={lags.std():.1f}  "
          f"min={lags.min()}  max={lags.max()}")
    hist, edges = np.histogram(lags, bins=11, range=(-args.max_lag, args.max_lag))
    for h, e in zip(hist, edges):
        bar = "#" * int(h * 60 / max(1, hist.max()))
        print(f"  {e:6.0f}: {bar} ({h})")


if __name__ == "__main__":
    main()
