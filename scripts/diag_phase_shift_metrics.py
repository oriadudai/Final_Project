"""Diagnostic: shift-corrected clinical metrics for saved ReHeartNet checkpoints.

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

Three modes:
  --checkpoint/--model-key/--fold   single checkpoint, optional --plot-windows
  --model-key --all-folds           every fold of one variant: per-fold metrics
                                     JSON + overlay PNG saved alongside its
                                     results (<results-dir>/<model_key>/diag_phase_shift/),
                                     plus an aggregate summary.json
  --compare-models                  every fold: one figure with 3 subplots
                                     (original/huber/clef), each showing GT vs
                                     raw vs shift-corrected pred for the same
                                     test window (cropped to 4 s for clef),
                                     saved to <results-dir>/figures/
  --model-key --train-vs-test       every fold of one variant: evaluate the
                                     saved checkpoint on its OWN (phase-aligned)
                                     training data as well as the (unaligned)
                                     test data, to tell apart "model never
                                     learned anything" from "model learned the
                                     aligned distribution but the train/test
                                     alignment mismatch hurts at test time".
                                     Saved to <results-dir>/<model_key>/
                                     diag_phase_shift/fold_NN_train_vs_test.json
                                     + train_vs_test_summary.json

Usage:
    python scripts/diag_phase_shift_metrics.py \
        --checkpoint results/comparison_reheartnet/reheartnet_clef/checkpoints/reheartnet_fold_00_best.pt \
        --model-key reheartnet_clef --fold 0 --plot-windows 4

    python scripts/diag_phase_shift_metrics.py --model-key reheartnet_clef --all-folds

    python scripts/diag_phase_shift_metrics.py --compare-models
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.config as config
from core.data_loader import build_group_fold, build_test_dataset, get_cv_splits
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
ALL_MODEL_KEYS = ["reheartnet_original", "reheartnet_huber", "reheartnet_clef"]
MODEL_LABELS = {
    "reheartnet_original": "Original (MSE)",
    "reheartnet_huber":    "Huber",
    "reheartnet_clef":     "Huber + CLEF (ours)",
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


def plot_examples(true_arr, pred_arr, shifted, lags, n, fs, save_path, title=None):
    """Save a PNG overlaying ground-truth ECG, raw prediction, and shift-corrected
    prediction for `n` evenly-spaced test windows."""
    n = min(n, len(true_arr))
    idx = np.linspace(0, len(true_arr) - 1, n, dtype=int)
    t = np.arange(true_arr.shape[1]) / fs

    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), squeeze=False)
    for row, i in enumerate(idx):
        ax = axes[row, 0]
        ax.plot(t, true_arr[i], color="#1f77b4", lw=1.2, label="Ground truth ECG", alpha=0.9)
        ax.plot(t, pred_arr[i], color="#d62728", lw=1.0, label="Predicted (raw)", alpha=0.7)
        ax.plot(t, shifted[i],  color="#2ca02c", lw=1.0, label="Predicted (shift-corrected)",
                alpha=0.85, linestyle="--")
        ax.set_title(f"window {i}  (best lag = {lags[i]:+d} samples = {lags[i] / fs * 1000:+.0f} ms)",
                      fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Amplitude (z)", fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title or "ECG reconstruction: ground truth vs predicted (raw & shift-corrected)",
                  fontsize=11, y=1.0)
    plt.tight_layout()
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


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


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _checkpoint_path(results_dir: str, model_key: str, fold: int) -> str:
    return os.path.join(results_dir, model_key, "checkpoints", f"reheartnet_fold_{fold:02d}_best.pt")


def _load_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    hidden_size = ckpt["hidden_size"]
    model = get_model("reheartnet", hidden_size=hidden_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _run_inference(model, test_ds, device, batch_size: int = 32):
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_true, all_pred = [], []
    with torch.no_grad():
        for ppg, ecg in loader:
            pred = model(ppg.to(device))
            all_true.append(ecg.squeeze(-1).numpy())
            all_pred.append(pred.squeeze(-1).cpu().numpy())
    return np.concatenate(all_true, axis=0), np.concatenate(all_pred, axis=0)


def _shift_correct_all(true_arr, pred_arr, max_lag):
    shifted = np.empty_like(pred_arr)
    lags = np.empty(len(pred_arr), dtype=int)
    for i in range(len(pred_arr)):
        shifted[i], lags[i] = best_shift(pred_arr[i], true_arr[i], max_lag)
    return shifted, lags


def _mean_ci(values: list) -> tuple:
    """(mean, 95% CI margin) for a list of fold values."""
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return float("nan"), 0.0
    mean   = float(np.mean(arr))
    margin = 1.96 * float(np.std(arr, ddof=1)) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, margin


def _maybe_classifier(model_key: str, args, device):
    """BCE classifier for reheartnet_clef only (see module docstring)."""
    if model_key != "reheartnet_clef":
        return None
    clef_size = args.clef_size
    if clef_size == "auto":
        clef_size = "medium" if device.type == "cuda" else "small"
    clef_path = args.clef_path or os.path.join(args.clef_dir, f"clef_{clef_size}.ckpt")
    print(f"Loading CLEF encoder ({clef_size}) from {clef_path} ...")
    clef_encoder = load_clef_encoder(clef_path, clef_size, device)
    return build_classifier(clef_encoder).to(device)


def _print_metric_table(model_key: str, fold: int, metrics_raw: dict, metrics_shift: dict) -> None:
    print(f"\n=== {model_key}  fold {fold} ===")
    print(f"{'metric':<16}{'raw':>12}{'shift-corrected':>18}")
    for key in ["rmse", "prd", "pearson_r", "emd", "ks_stat", "ks_pvalue", "beat_timing_mae", "bce"]:
        if key not in metrics_raw:
            continue
        print(f"{key:<16}{metrics_raw[key]:>12.4f}{metrics_shift[key]:>18.4f}")


# ------------------------------------------------------------------
# Modes
# ------------------------------------------------------------------

def run_single(args, device) -> None:
    model, ckpt = _load_model(args.checkpoint, device)
    print(f"hidden_size from checkpoint: {ckpt['hidden_size']}")
    print(f"checkpoint fold: {ckpt['fold']}  epoch: {ckpt['epoch']}  val_loss: {ckpt['val_loss']:.5f}")

    all_subjects = get_all_record_names()
    splits = get_cv_splits(all_subjects, n_splits=8,
                            save_path=os.path.join("results", "_diag_fold_assignments.json"))
    _, test_subs = splits[args.fold]
    print(f"fold {args.fold} test subjects: {test_subs}")

    test_ds = build_test_dataset(test_subs, **PREPROC[args.model_key])
    true_arr, pred_arr = _run_inference(model, test_ds, device)
    print(f"test windows: {true_arr.shape[0]}  window length: {true_arr.shape[1]}")

    classifier = _maybe_classifier(args.model_key, args, device)
    if classifier is None:
        print("BCE skipped (4s-window model -- see module docstring).")

    shifted, lags = _shift_correct_all(true_arr, pred_arr, args.max_lag)

    metrics_raw   = compute_all_metrics(true_arr, pred_arr, config.FS, classifier, device)
    metrics_shift = compute_all_metrics(true_arr, shifted,  config.FS, classifier, device)
    _print_metric_table(args.model_key, args.fold, metrics_raw, metrics_shift)

    print(f"\nLag distribution (samples, +/-{args.max_lag}):")
    print(f"  mean={lags.mean():.1f}  std={lags.std():.1f}  "
          f"min={lags.min()}  max={lags.max()}")
    hist, edges = np.histogram(lags, bins=11, range=(-args.max_lag, args.max_lag))
    for h, e in zip(hist, edges):
        bar = "#" * int(h * 60 / max(1, hist.max()))
        print(f"  {e:6.0f}: {bar} ({h})")

    if args.plot_windows > 0:
        plot_path = args.plot_out or os.path.join(
            "results", f"diag_phase_shift_{args.model_key}_fold{args.fold:02d}.png"
        )
        plot_examples(true_arr, pred_arr, shifted, lags, args.plot_windows, config.FS, plot_path)
        print(f"\nSaved comparison plot: {plot_path}")


def run_all_folds(args, device) -> None:
    model_key = args.model_key
    out_dir = os.path.join(args.results_dir, model_key, "diag_phase_shift")
    os.makedirs(out_dir, exist_ok=True)

    classifier = _maybe_classifier(model_key, args, device)
    if classifier is None:
        print("BCE skipped (4s-window model -- see module docstring).")

    all_subjects = get_all_record_names()
    splits = get_cv_splits(all_subjects, n_splits=args.n_folds,
                            save_path=os.path.join("results", "_diag_fold_assignments.json"))

    n_plot = args.plot_windows if args.plot_windows > 0 else 3
    per_fold_raw, per_fold_shift = [], []

    for fold in range(args.n_folds):
        ckpt_path = _checkpoint_path(args.results_dir, model_key, fold)
        if not os.path.exists(ckpt_path):
            print(f"  fold {fold:02d}: checkpoint not found ({ckpt_path}), skipping")
            continue

        model, ckpt = _load_model(ckpt_path, device)
        _, test_subs = splits[fold]
        test_ds = build_test_dataset(test_subs, **PREPROC[model_key])
        true_arr, pred_arr = _run_inference(model, test_ds, device)
        shifted, lags = _shift_correct_all(true_arr, pred_arr, args.max_lag)

        metrics_raw   = compute_all_metrics(true_arr, pred_arr, config.FS, classifier, device)
        metrics_shift = compute_all_metrics(true_arr, shifted,  config.FS, classifier, device)
        per_fold_raw.append(metrics_raw)
        per_fold_shift.append(metrics_shift)

        fold_result = {
            "fold":                fold,
            "checkpoint_epoch":    ckpt["epoch"],
            "checkpoint_val_loss": ckpt["val_loss"],
            "n_windows":           int(true_arr.shape[0]),
            "metrics_raw":             metrics_raw,
            "metrics_shift_corrected": metrics_shift,
            "lag_samples": {
                "mean": float(lags.mean()), "std": float(lags.std()),
                "min": int(lags.min()), "max": int(lags.max()),
            },
        }
        with open(os.path.join(out_dir, f"fold_{fold:02d}_metrics.json"), "w") as f:
            json.dump(fold_result, f, indent=2)

        plot_path = os.path.join(out_dir, f"fold_{fold:02d}_recon.png")
        plot_examples(true_arr, pred_arr, shifted, lags, n_plot, config.FS, plot_path,
                      title=f"{model_key}  fold {fold:02d}: ground truth vs predicted "
                            f"(raw & shift-corrected)")

        print(f"  fold {fold:02d}: rmse {metrics_raw['rmse']:.4f} -> {metrics_shift['rmse']:.4f}"
              f"  prd {metrics_raw['prd']:.2f} -> {metrics_shift['prd']:.2f}"
              f"  lag mean={lags.mean():+.1f} samples")

    if not per_fold_raw:
        print("No checkpoints found -- nothing to summarize.")
        return

    summary = {"n_folds": len(per_fold_raw), "raw": {}, "shift_corrected": {}}
    for key in per_fold_raw[0]:
        for label, src in (("raw", per_fold_raw), ("shift_corrected", per_fold_shift)):
            mean, margin = _mean_ci([m.get(key, float("nan")) for m in src])
            summary[label][key] = {"mean": mean, "ci95": margin}

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved aggregate summary: {summary_path}")


def run_train_vs_test(args, device) -> None:
    """For one ReHeartNet variant, every fold: evaluate the saved checkpoint on
    its OWN (phase-aligned) training data as well as the (unaligned) test data.

    Disambiguates two failure modes when test metrics look degenerate:
      - train metrics ALSO degenerate -> the model never learned a useful
        PPG->ECG mapping (e.g. training collapse from lr=1e-2 instability),
        independent of any train/test alignment mismatch.
      - train metrics good, test metrics degenerate -> consistent with the
        train-only phase alignment causing a real train/test distribution
        shift (see module docstring).
    """
    model_key = args.model_key
    out_dir = os.path.join(args.results_dir, model_key, "diag_phase_shift")
    os.makedirs(out_dir, exist_ok=True)

    classifier = _maybe_classifier(model_key, args, device)

    all_subjects = get_all_record_names()
    splits = get_cv_splits(all_subjects, n_splits=args.n_folds,
                            save_path=os.path.join("results", "_diag_fold_assignments.json"))

    pp = PREPROC[model_key]
    per_fold = []

    for fold in range(args.n_folds):
        ckpt_path = _checkpoint_path(args.results_dir, model_key, fold)
        if not os.path.exists(ckpt_path):
            print(f"  fold {fold:02d}: checkpoint not found ({ckpt_path}), skipping")
            continue

        model, ckpt = _load_model(ckpt_path, device)
        train_subs, test_subs = splits[fold]
        # Same train-only phase-aligned data the model was actually fit on.
        train_ds, test_ds = build_group_fold(train_subs, test_subs, apply_align=True, **pp)

        train_true, train_pred = _run_inference(model, train_ds, device)
        test_true, test_pred   = _run_inference(model, test_ds, device)

        train_metrics = compute_all_metrics(train_true, train_pred, config.FS, classifier, device)
        test_raw      = compute_all_metrics(test_true, test_pred, config.FS, classifier, device)
        shifted, lags = _shift_correct_all(test_true, test_pred, args.max_lag)
        test_shift    = compute_all_metrics(test_true, shifted, config.FS, classifier, device)

        result = {
            "fold":                fold,
            "checkpoint_epoch":    ckpt["epoch"],
            "checkpoint_val_loss": ckpt["val_loss"],
            "n_train_windows":     int(train_true.shape[0]),
            "n_test_windows":      int(test_true.shape[0]),
            "train_metrics":                train_metrics,
            "test_metrics_raw":             test_raw,
            "test_metrics_shift_corrected": test_shift,
        }
        per_fold.append(result)
        with open(os.path.join(out_dir, f"fold_{fold:02d}_train_vs_test.json"), "w") as f:
            json.dump(result, f, indent=2)

        print(f"  fold {fold:02d} (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.5f}): "
              f"train rmse={train_metrics['rmse']:.4f} prd={train_metrics['prd']:.2f}  |  "
              f"test rmse={test_raw['rmse']:.4f}->{test_shift['rmse']:.4f} "
              f"prd={test_raw['prd']:.2f}->{test_shift['prd']:.2f}")

    if not per_fold:
        print("No checkpoints found -- nothing to summarize.")
        return

    summary = {"n_folds": len(per_fold), "train": {}, "test_raw": {}, "test_shift_corrected": {}}
    for key in per_fold[0]["train_metrics"]:
        for label, src_key in (("train", "train_metrics"),
                                ("test_raw", "test_metrics_raw"),
                                ("test_shift_corrected", "test_metrics_shift_corrected")):
            mean, margin = _mean_ci([r[src_key].get(key, float("nan")) for r in per_fold])
            summary[label][key] = {"mean": mean, "ci95": margin}

    summary_path = os.path.join(out_dir, "train_vs_test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_path}")


def run_compare_models(args, device) -> None:
    out_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    all_subjects = get_all_record_names()
    splits = get_cv_splits(all_subjects, n_splits=args.n_folds,
                            save_path=os.path.join("results", "_diag_fold_assignments.json"))

    crop_samples = int(round(4.0 * config.FS))  # 4 s @ 125 Hz -- crops the 10 s CLEF window

    for fold in range(args.n_folds):
        _, test_subs = splits[fold]
        fig, axes = plt.subplots(3, 1, figsize=(10, 9))

        for ax, model_key in zip(axes, ALL_MODEL_KEYS):
            ckpt_path = _checkpoint_path(args.results_dir, model_key, fold)
            if not os.path.exists(ckpt_path):
                ax.set_title(f"{MODEL_LABELS[model_key]} -- checkpoint not found", fontsize=9)
                ax.axis("off")
                continue

            model, _ = _load_model(ckpt_path, device)
            test_ds = build_test_dataset(test_subs[:1], **PREPROC[model_key])
            ppg0, ecg0 = test_ds[0]
            with torch.no_grad():
                pred0 = model(ppg0.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            true0 = ecg0.squeeze().numpy()
            shifted0, lag0 = best_shift(pred0, true0, args.max_lag)

            true_c  = true0[:crop_samples]
            pred_c  = pred0[:crop_samples]
            shift_c = shifted0[:crop_samples]
            t = np.arange(len(true_c)) / config.FS

            ax.plot(t, true_c,  color="#1f77b4", lw=1.2, alpha=0.9, label="Ground truth ECG")
            ax.plot(t, pred_c,  color="#d62728", lw=1.0, alpha=0.7, label="Predicted (raw)")
            ax.plot(t, shift_c, color="#2ca02c", lw=1.0, alpha=0.85, linestyle="--",
                    label="Predicted (shift-corrected)")
            ax.set_title(f"{MODEL_LABELS[model_key]}  (lag = {lag0:+d} samples = "
                          f"{lag0 / config.FS * 1000:+.0f} ms)", fontsize=9)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Amplitude (z)", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Fold {fold:02d}  --  test subject {test_subs[0]}: ground truth vs "
                      f"predicted (raw & shift-corrected), first 4 s", fontsize=11, y=1.0)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"diag_phase_shift_compare_fold{fold:02d}.png")
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None,
                     help="(single mode) path to a specific checkpoint")
    ap.add_argument("--model-key", type=str, default=None, choices=list(PREPROC))
    ap.add_argument("--fold", type=int, default=None, help="(single mode) fold index")
    ap.add_argument("--max-lag", type=int, default=125)
    ap.add_argument("--clef-path", type=str, default=None,
                     help="Path to CLEF .ckpt for BCE (CLEF model only). "
                          "If omitted, auto-constructs from --clef-dir/--clef-size.")
    ap.add_argument("--clef-size", type=str, default="auto",
                     choices=["auto", "small", "medium", "large"])
    ap.add_argument("--clef-dir", type=str, default=config.CLEF_CHECKPOINT_DIR)
    ap.add_argument("--plot-windows", type=int, default=0,
                     help="If >0, save a PNG comparing ground truth / raw pred / "
                          "shift-corrected pred for this many evenly-spaced windows. "
                          "(--all-folds defaults to 3 if not set.)")
    ap.add_argument("--plot-out", type=str, default=None,
                     help="(single mode) Output PNG path (default: "
                          "results/diag_phase_shift_<model_key>_fold<NN>.png)")
    ap.add_argument("--all-folds", action="store_true",
                     help="Compute raw-vs-shift-corrected metrics + overlay plot for "
                          "every fold of --model-key, saved to "
                          "<results-dir>/<model_key>/diag_phase_shift/, plus summary.json.")
    ap.add_argument("--compare-models", action="store_true",
                     help="For every fold, save one figure with 3 subplots (original/"
                          "huber/clef), each showing GT vs raw vs shift-corrected pred "
                          "for the same test window (cropped to 4s for clef), to "
                          "<results-dir>/figures/.")
    ap.add_argument("--train-vs-test", action="store_true",
                     help="For every fold of --model-key, evaluate the saved checkpoint "
                          "on its own (phase-aligned) training data as well as the "
                          "(unaligned) test data. Saved to <results-dir>/<model_key>/"
                          "diag_phase_shift/fold_NN_train_vs_test.json + "
                          "train_vs_test_summary.json")
    ap.add_argument("--results-dir", type=str, default=os.path.join("results", "comparison_reheartnet"),
                     help="Base directory holding <model_key>/checkpoints/ "
                          "(default: results/comparison_reheartnet)")
    ap.add_argument("--n-folds", type=int, default=8)
    args = ap.parse_args()

    device = config.DEVICE

    if args.compare_models:
        run_compare_models(args, device)
    elif args.train_vs_test:
        if args.model_key is None:
            raise SystemExit("--train-vs-test requires --model-key")
        run_train_vs_test(args, device)
    elif args.all_folds:
        if args.model_key is None:
            raise SystemExit("--all-folds requires --model-key")
        run_all_folds(args, device)
    else:
        if args.checkpoint is None or args.model_key is None or args.fold is None:
            raise SystemExit("single-checkpoint mode requires --checkpoint, --model-key, "
                              "and --fold (or use --all-folds / --compare-models)")
        run_single(args, device)


if __name__ == "__main__":
    main()
