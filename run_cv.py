"""Main 8-fold group cross-validation script for ReHeartNet and baselines.

Run the proposed model:
    python run_cv.py --clef-path models/clef/clef_small.ckpt

Run all models (ablation table):
    python run_cv.py --clef-path models/clef/clef_small.ckpt --all-models

Run a specific baseline:
    python run_cv.py --clef-path models/clef/clef_small.ckpt --model bilstm

Resume from a crashed fold:
    python run_cv.py --clef-path models/clef/clef_small.ckpt --resume-fold 3

Dry-run (2 folds, 2 epochs) for sanity check:
    python run_cv.py --clef-path models/clef/clef_small.ckpt --dry-run --no-wandb
"""

import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import core.config as config
from core.data_loader import build_group_fold, get_cv_splits, split_train_val
from core.evaluate import evaluate_fold
from core.losses.composite_loss import load_clef_encoder
from core.metrics.clinical_metrics import extract_rr_intervals
from core.models.ptbxl_classifier import get_ptbxl_classifier
from core.train import train_fold
from core.visualization.plots import (
    plot_fold_metrics_bar,
    plot_loss_curves,
    plot_metric_boxplots,
    plot_reconstruction_samples,
    plot_rr_distributions,
    save_results_summary,
)
from src.preprocessing import get_all_record_names

_ALL_MODELS = ["linear", "lstm", "bilstm", "reheartnet"]
_METRICS    = ["prd", "pearson_r", "bce", "emd", "ks_stat", "beat_timing_mae"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ReHeartNet 8-fold CV training and evaluation")
    p.add_argument("--clef-path",        type=str, required=True, help="Path to CLEF .ckpt")
    p.add_argument("--clef-size",        type=str, default="small", choices=["small","medium","large"])
    p.add_argument("--classifier-path",  type=str, default=None, help="Path to PTB-XL classifier .pt")
    p.add_argument("--output-dir",       type=str, default="results")
    p.add_argument("--n-folds",          type=int, default=8)
    p.add_argument("--epochs",           type=int, default=None)
    p.add_argument("--batch-size",       type=int, default=None)
    p.add_argument("--resume-fold",      type=int, default=0, help="Start from fold index (0-based)")
    p.add_argument("--dry-run",          action="store_true", help="2 folds, 2 epochs each")
    p.add_argument("--no-wandb",         action="store_true")
    p.add_argument(
        "--model", type=str, default="reheartnet",
        choices=_ALL_MODELS,
        help="Architecture to train.",
    )
    p.add_argument(
        "--all-models", action="store_true",
        help="Run all four models sequentially and save a comparison summary.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Hyperparameter loading
# ---------------------------------------------------------------------------

def _load_best_hyperparams(output_dir: str) -> dict:
    path = os.path.join(output_dir, "best_hyperparams.json")
    if os.path.exists(path):
        with open(path) as f:
            params = json.load(f)
        print(f"Loaded hyperparameters from {path}")
        return params
    print("No best_hyperparams.json found — using config defaults.")
    return {}


# ---------------------------------------------------------------------------
# Single-model CV run
# ---------------------------------------------------------------------------

def _run_single_model(
    args: argparse.Namespace,
    device: torch.device,
    model_name: str,
    clef_encoder: torch.nn.Module = None,
    ptbxl_clf:    torch.nn.Module = None,
    splits: list = None,
) -> list:
    """Run the full CV loop for one model. Returns list of per-fold metric dicts."""

    best = _load_best_hyperparams(args.output_dir)

    lr              = float(best.get("lr",              config.LEARNING_RATE))
    lambda_clinical = float(best.get("lambda_clinical", config.LAMBDA_CLINICAL))
    huber_delta     = float(best.get("huber_delta",     config.HUBER_DELTA))
    batch_size      = int(best.get("batch_size",        config.BATCH_SIZE))
    hidden_size     = int(best.get("hidden_size",       config.HIDDEN_SIZE))
    epochs          = int(best.get("epochs",            config.EPOCHS))

    if args.epochs     is not None: epochs     = args.epochs
    if args.batch_size is not None: batch_size = args.batch_size
    if args.dry_run:
        epochs = 2

    print(f"[{model_name}] lr={lr}, λ={lambda_clinical}, δ={huber_delta}, "
          f"bs={batch_size}, H={hidden_size}, epochs={epochs}")

    # Build shared frozen models once if not provided
    if clef_encoder is None:
        print(f"Loading CLEF encoder ({args.clef_size}) ...")
        clef_encoder = load_clef_encoder(args.clef_path, args.clef_size, device)
    if ptbxl_clf is None:
        ptbxl_clf = get_ptbxl_classifier(args.classifier_path).to(device)

    # Generate CV splits once if not provided
    if splits is None:
        all_subjects = get_all_record_names()
        splits = get_cv_splits(
            all_subjects,
            n_splits  = args.n_folds,
            save_path = os.path.join(args.output_dir, "fold_assignments.json"),
        )
        if args.dry_run:
            splits = splits[:2]

    # Resume support
    partial_path  = os.path.join(args.output_dir, f"fold_metrics_{model_name}_partial.json")
    fold_metrics: list = []
    if args.resume_fold > 0 and os.path.exists(partial_path):
        with open(partial_path) as f:
            fold_metrics = json.load(f)
        print(f"Resuming {model_name} from fold {args.resume_fold}.")

    # -----------------------------------------------------------------------
    # Main CV loop
    # -----------------------------------------------------------------------
    for fold_idx, (train_subs, test_subs) in enumerate(
        tqdm(splits, desc=f"CV [{model_name}]", unit="fold")
    ):
        if fold_idx < args.resume_fold:
            continue

        print(f"\n{'='*55}")
        print(f"[{model_name.upper()}] Fold {fold_idx:02d}/{len(splits)-1}  test: {test_subs}")
        print(f"{'='*55}")

        train_ds, test_ds = build_group_fold(train_subs, test_subs, apply_align=True)
        train_inner, val_ds = split_train_val(train_ds, val_fraction=0.1)

        train_loader = DataLoader(train_inner, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,      batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        test_loader  = DataLoader(test_ds,     batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        print(f"  Train: {len(train_inner)} | Val: {len(val_ds)} | Test: {len(test_ds)} windows")

        model, history = train_fold(
            fold_idx        = fold_idx,
            fold_subjects   = test_subs,
            train_loader    = train_loader,
            val_loader      = val_loader,
            clef_encoder    = clef_encoder,
            device          = device,
            epochs          = epochs,
            lr              = lr,
            lambda_clinical = lambda_clinical,
            huber_delta     = huber_delta,
            hidden_size     = hidden_size,
            checkpoint_dir  = config.CHECKPOINT_DIR,
            use_wandb       = not args.no_wandb,
            model_name      = model_name,
        )

        metrics = evaluate_fold(model, test_loader, ptbxl_clf, device, clef_encoder=clef_encoder)
        metrics["fold"]       = fold_idx
        metrics["subjects"]   = test_subs
        metrics["model_name"] = model_name
        fold_metrics.append(metrics)

        print(f"  PRD={metrics['prd']:.3f}  r={metrics['pearson_r']:.3f}  "
              f"BCE={metrics['bce']:.4f}  EMD={metrics['emd']:.4f}  "
              f"KS={metrics['ks_stat']:.3f}  beat-MAE={metrics['beat_timing_mae']:.4f}s")

        # Per-fold figures (prefixed with model name)
        _save_fold_figures(model, test_loader, test_subs, history,
                           fold_idx, device, args.output_dir, model_name)

        with open(partial_path, "w") as f:
            json.dump(fold_metrics, f, indent=2, default=str)

    # Aggregate figures + summary for this model
    fig_prefix = os.path.join(args.output_dir, "figures")
    for metric_name in _METRICS:
        plot_fold_metrics_bar(
            fold_metrics, metric_name,
            save_path=os.path.join(fig_prefix, f"metric_bar_{model_name}_{metric_name}.png"),
        )
    plot_metric_boxplots(
        fold_metrics,
        save_path=os.path.join(fig_prefix, f"metric_boxplots_{model_name}.png"),
    )
    save_results_summary(
        fold_metrics,
        save_path=os.path.join(args.output_dir, f"summary_{model_name}.json"),
    )

    return fold_metrics


# ---------------------------------------------------------------------------
# Multi-model comparison helpers
# ---------------------------------------------------------------------------

def _save_comparison_summary(
    all_model_metrics: dict,   # {model_name: [fold_metric_dicts]}
    output_dir: str,
) -> None:
    """Save a single JSON with mean ± CI for all models side-by-side."""
    comparison = {}
    for model_name, fold_metrics in all_model_metrics.items():
        model_summary = {}
        for metric in _METRICS:
            vals = [m[metric] for m in fold_metrics if not np.isnan(m.get(metric, float("nan")))]
            if not vals:
                model_summary[metric] = {"mean": None, "ci95_low": None, "ci95_high": None}
                continue
            arr    = np.array(vals)
            mean   = float(np.mean(arr))
            margin = 1.96 * float(np.std(arr, ddof=1)) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
            model_summary[metric] = {
                "mean":      mean,
                "ci95_low":  mean - margin,
                "ci95_high": mean + margin,
            }
        comparison[model_name] = model_summary

    path = os.path.join(output_dir, "comparison_summary.json")
    with open(path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nComparison summary saved to {path}")

    # Print table to console
    print(f"\n{'Model':<20}", end="")
    for m in _METRICS:
        print(f"  {m:<18}", end="")
    print()
    print("-" * (20 + 20 * len(_METRICS)))
    for model_name, model_summary in comparison.items():
        print(f"{model_name:<20}", end="")
        for m in _METRICS:
            s = model_summary[m]
            if s["mean"] is not None:
                print(f"  {s['mean']:.3f}±{s['mean']-s['ci95_low']:.3f}   ", end="")
            else:
                print(f"  {'N/A':<18}", end="")
        print()


def _plot_comparison(
    all_model_metrics: dict,
    output_dir: str,
) -> None:
    """Bar chart comparing all models on each metric (mean ± 95% CI)."""
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    model_names  = list(all_model_metrics.keys())
    display_names = {"linear": "Linear\nReg.", "lstm": "S-LSTM", "bilstm": "P-BiLSTM", "reheartnet": "ReHeartNet\n(ours)"}

    for metric in _METRICS:
        means, lows, highs = [], [], []
        for model_name in model_names:
            vals = [m[metric] for m in all_model_metrics[model_name]
                    if not np.isnan(m.get(metric, float("nan")))]
            if not vals:
                means.append(0); lows.append(0); highs.append(0)
                continue
            arr = np.array(vals)
            mu  = float(np.mean(arr))
            ci  = 1.96 * float(np.std(arr, ddof=1)) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
            means.append(mu); lows.append(mu - ci); highs.append(mu + ci)

        x      = np.arange(len(model_names))
        colors = ["#aec6cf", "#b5ead7", "#ffdac1", "#ff9aa2"]
        errs   = [[m - l for m, l in zip(means, lows)],
                  [h - m for m, h in zip(means, highs)]]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(x, means, yerr=errs, capsize=5, color=colors[:len(model_names)], edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([display_names.get(n, n) for n in model_names], fontsize=9)
        ax.set_ylabel(metric, fontsize=9)
        ax.set_title(f"Model comparison — {metric}", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(output_dir, "figures", f"comparison_{metric}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Comparison figures saved to {os.path.join(output_dir, 'figures')}/comparison_*.png")


# ---------------------------------------------------------------------------
# Per-fold visualizations
# ---------------------------------------------------------------------------

def _save_fold_figures(model, test_loader, test_subs, history,
                       fold_idx, device, output_dir, model_name="reheartnet"):
    fig_dir = os.path.join(output_dir, "figures")

    plot_loss_curves(
        history["train"], history["val"], fold_idx,
        save_path=os.path.join(fig_dir, f"loss_curves_{model_name}_fold{fold_idx:02d}.png"),
    )

    model.eval()
    with torch.no_grad():
        ppg_batch, ecg_batch = next(iter(test_loader))
        pred_batch = model(ppg_batch.to(device)).cpu()

    true_np = ecg_batch.squeeze(-1).numpy()
    pred_np = pred_batch.squeeze(-1).numpy()

    plot_reconstruction_samples(
        true_np, pred_np,
        subject_ids=[test_subs[min(i, len(test_subs)-1)] for i in range(min(4, len(true_np)))],
        fold_idx=fold_idx,
        save_path=os.path.join(fig_dir, f"reconstruction_{model_name}_fold{fold_idx:02d}.png"),
    )

    all_true, all_pred = [], []
    with torch.no_grad():
        for ppg, ecg in test_loader:
            pred = model(ppg.to(device)).cpu()
            all_true.append(ecg.squeeze(-1).numpy())
            all_pred.append(pred.squeeze(-1).numpy())
    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rr_true = extract_rr_intervals(all_true)
        rr_pred = extract_rr_intervals(all_pred)

    plot_rr_distributions(
        rr_true, rr_pred,
        subject_id=f"{model_name}_fold{fold_idx:02d}",
        fold_idx=fold_idx,
        save_path=os.path.join(fig_dir, f"rr_dist_{model_name}_fold{fold_idx:02d}.png"),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    device = config.DEVICE
    print(f"Device: {device}")

    # Build shared frozen models and splits ONCE (reused across all model runs)
    print(f"Loading CLEF encoder ({args.clef_size}) ...")
    clef_encoder = load_clef_encoder(args.clef_path, args.clef_size, device)
    ptbxl_clf    = get_ptbxl_classifier(args.classifier_path).to(device)

    all_subjects = get_all_record_names()
    n_folds      = 2 if args.dry_run else args.n_folds
    splits = get_cv_splits(
        all_subjects,
        n_splits  = n_folds,
        save_path = os.path.join(args.output_dir, "fold_assignments.json"),
    )

    if args.all_models:
        all_model_metrics = {}
        for model_name in _ALL_MODELS:
            print(f"\n{'#'*60}\nModel: {model_name.upper()}\n{'#'*60}")
            all_model_metrics[model_name] = _run_single_model(
                args, device, model_name,
                clef_encoder=clef_encoder, ptbxl_clf=ptbxl_clf, splits=splits,
            )
        _save_comparison_summary(all_model_metrics, args.output_dir)
        _plot_comparison(all_model_metrics, args.output_dir)
    else:
        _run_single_model(
            args, device, args.model,
            clef_encoder=clef_encoder, ptbxl_clf=ptbxl_clf, splits=splits,
        )

    print("\nDone. Results in:", args.output_dir)


if __name__ == "__main__":
    main()
