"""Compare original ReHeartNet vs. our CLEF-enhanced ReHeartNet.

Both models share the same DC-BiLSTM architecture (same hyperparameters,
same dataset splits, same evaluation protocol) and differ only in the
training objective:

  - Original ReHeartNet : Huber loss only          (lambda_clinical = 0)
  - Ours (+ CLEF)       : Huber + CLEF feature loss (lambda_clinical from Optuna)

This script runs the full 8-fold CV for both models and produces:
  - results/comparison_reheartnet/summary_{model}.json   per-model summary
  - results/comparison_reheartnet/comparison.json        side-by-side
  - results/comparison_reheartnet/figures/               all plots

Usage:
    python scripts/compare_reheartnet.py --clef-path models/clef/clef_small.ckpt
    python scripts/compare_reheartnet.py --clef-path models/clef/clef_small.ckpt --dry-run
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.config as config
from core.data_loader import build_group_fold, get_cv_splits, split_train_val
from core.evaluate import evaluate_fold
from core.losses.composite_loss import load_clef_encoder
from core.metrics.clinical_metrics import extract_rr_intervals
from core.models.ptbxl_classifier import get_ptbxl_classifier
from core.train import train_fold
from core.visualization.plots import (
    plot_loss_curves,
    plot_reconstruction_samples,
    plot_rr_distributions,
    save_results_summary,
)
from src.preprocessing import get_all_record_names

# ──────────────────────────────────────────────────────────────────────────────
# Three model variants — same DC-BiLSTM architecture, different training loss
#
#  original  : MSE only          → faithful replica of Ye et al. (2023)
#  huber     : Huber only (λ=0)  → isolates loss-function change
#  clef      : Huber + CLEF      → our full method
# ──────────────────────────────────────────────────────────────────────────────
MODELS = {
    "reheartnet_mse":   {"label": "ReHeartNet (MSE)",         "loss_type": "mse"},
    "reheartnet_huber": {"label": "ReHeartNet + Huber",       "loss_type": "huber"},
    "reheartnet_clef":  {"label": "ReHeartNet + CLEF (ours)", "loss_type": "clef"},
}

METRICS = ["prd", "pearson_r", "bce", "emd", "ks_stat", "beat_timing_mae"]
METRIC_LABELS = {
    "prd":             "PRD (%)",
    "pearson_r":       "Pearson r",
    "bce":             "BCE",
    "emd":             "EMD (s)",
    "ks_stat":         "KS statistic",
    "beat_timing_mae": "Beat MAE (s)",
}
# For each metric: True = lower is better, False = higher is better
LOWER_BETTER = {
    "prd": True, "pearson_r": False, "bce": True,
    "emd": True, "ks_stat": True, "beat_timing_mae": True,
}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare original ReHeartNet vs. CLEF-enhanced ReHeartNet")
    p.add_argument("--clef-path",       type=str, required=True)
    p.add_argument("--clef-size",       type=str, default="small", choices=["small", "medium", "large"])
    p.add_argument("--classifier-path", type=str, default=None)
    p.add_argument("--output-dir",      type=str, default=os.path.join("results", "comparison_reheartnet"))
    p.add_argument("--n-folds",         type=int, default=8)
    p.add_argument("--epochs",          type=int, default=None)
    p.add_argument("--batch-size",      type=int, default=None)
    p.add_argument("--resume",          action="store_true", help="Resume from existing partial results")
    p.add_argument("--dry-run",         action="store_true", help="2 folds, 2 epochs each")
    p.add_argument("--no-wandb",        action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Single-model CV loop
# ──────────────────────────────────────────────────────────────────────────────

def _run_model(
    model_key:    str,
    model_cfg:    dict,
    args:         argparse.Namespace,
    device:       torch.device,
    clef_encoder: torch.nn.Module,
    ptbxl_clf:    torch.nn.Module,
    splits:       list,
    best_params:  dict,
) -> list:
    """Run 8-fold CV for one model variant. Returns list of fold metric dicts."""

    epochs      = args.epochs     or int(best_params.get("epochs",     config.EPOCHS))
    batch_size  = args.batch_size or int(best_params.get("batch_size", config.BATCH_SIZE))
    hidden_size =                    int(best_params.get("hidden_size", config.HIDDEN_SIZE))
    lr          =                  float(best_params.get("lr",          config.LEARNING_RATE))
    huber_delta =                  float(best_params.get("huber_delta", config.HUBER_DELTA))
    lambda_clinical = float(best_params.get("lambda_clinical", config.LAMBDA_CLINICAL))

    loss_type = model_cfg["loss_type"]

    if args.dry_run:
        epochs = 2

    label = model_cfg["label"]
    print(f"\n  loss={loss_type}  |  lr={lr}  |  H={hidden_size}  |  epochs={epochs}")

    partial_path = os.path.join(args.output_dir, f"partial_{model_key}.json")
    fold_metrics: list = []
    resume_from = 0
    if args.resume and os.path.exists(partial_path):
        with open(partial_path) as f:
            fold_metrics = json.load(f)
        resume_from = len(fold_metrics)
        print(f"  Resuming {label} from fold {resume_from}.")

    for fold_idx, (train_subs, test_subs) in enumerate(
        tqdm(splits, desc=label, unit="fold")
    ):
        if fold_idx < resume_from:
            continue

        train_ds, test_ds = build_group_fold(train_subs, test_subs, apply_align=True)
        train_inner, val_ds = split_train_val(train_ds, val_fraction=0.1)

        train_loader = DataLoader(train_inner, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,      batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        test_loader  = DataLoader(test_ds,     batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

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
            checkpoint_dir  = os.path.join(args.output_dir, "checkpoints"),
            use_wandb       = not args.no_wandb,
            model_name      = "reheartnet",   # same DC-BiLSTM architecture for all three
            loss_type       = loss_type,
        )

        metrics = evaluate_fold(model, test_loader, ptbxl_clf, device, clef_encoder=clef_encoder)
        metrics.update({"fold": fold_idx, "subjects": test_subs, "model": model_key})
        fold_metrics.append(metrics)

        print(
            f"    [{label}] fold {fold_idx:02d} — "
            f"PRD={metrics['prd']:.3f}  r={metrics['pearson_r']:.3f}  "
            f"BCE={metrics['bce']:.4f}  EMD={metrics['emd']:.4f}  "
            f"KS={metrics['ks_stat']:.3f}  beat-MAE={metrics['beat_timing_mae']:.4f}s"
        )

        # Per-fold figures
        fig_dir = os.path.join(args.output_dir, "figures")
        plot_loss_curves(
            history["train"], history["val"], fold_idx,
            save_path=os.path.join(fig_dir, f"loss_{model_key}_fold{fold_idx:02d}.png"),
        )
        _save_recon_figure(model, test_loader, test_subs, device,
                           fold_idx, model_key, fig_dir)

        with open(partial_path, "w") as f:
            json.dump(fold_metrics, f, indent=2, default=str)

    save_results_summary(
        fold_metrics,
        save_path=os.path.join(args.output_dir, f"summary_{model_key}.json"),
    )
    return fold_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Comparison output
# ──────────────────────────────────────────────────────────────────────────────

def _compute_mean_ci(values: list) -> tuple:
    """(mean, ci_margin) for a list of fold values."""
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return float("nan"), 0.0
    mean   = float(np.mean(arr))
    margin = 1.96 * float(np.std(arr, ddof=1)) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, margin


def _save_comparison_json(
    all_results: dict,  # {model_key: [fold_metric_dicts]}
    output_dir: str,
) -> dict:
    """Build and save side-by-side summary. Returns the summary dict."""
    summary = {}
    for model_key, fold_metrics in all_results.items():
        summary[model_key] = {"label": MODELS[model_key]["label"]}
        for metric in METRICS:
            vals = [m.get(metric, float("nan")) for m in fold_metrics]
            mean, margin = _compute_mean_ci(vals)
            summary[model_key][metric] = {
                "mean":      mean,
                "ci_margin": margin,
                "per_fold":  vals,
            }

    path = os.path.join(output_dir, "comparison.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nComparison saved → {path}")
    return summary


def _print_comparison_table(summary: dict) -> None:
    """Print a formatted comparison table to stdout."""
    models = list(summary.keys())
    col_w  = 26

    header = f"{'Metric':<22}"
    for mk in models:
        header += f"  {summary[mk]['label']:^{col_w}}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for metric in METRICS:
        row = f"{METRIC_LABELS[metric]:<22}"
        best_mean = None
        for mk in models:
            m = summary[mk][metric]["mean"]
            if not np.isnan(m):
                if best_mean is None:
                    best_mean = m
                elif LOWER_BETTER[metric] and m < best_mean:
                    best_mean = m
                elif not LOWER_BETTER[metric] and m > best_mean:
                    best_mean = m

        for mk in models:
            m  = summary[mk][metric]["mean"]
            ci = summary[mk][metric]["ci_margin"]
            if np.isnan(m):
                cell = "N/A"
            else:
                cell = f"{m:.4f} ± {ci:.4f}"
                if m == best_mean:
                    cell = f"**{cell}**"
            row += f"  {cell:^{col_w}}"
        print(row)
    print("=" * len(header))


def _plot_side_by_side(summary: dict, output_dir: str) -> None:
    """One figure per metric: side-by-side bars with CI for both models."""
    fig_dir = os.path.join(output_dir, "figures")
    models  = list(summary.keys())
    labels  = [summary[mk]["label"] for mk in models]
    colors  = ["#7fb3d3", "#a8d5a2", "#e8906a"]   # blue=MSE, green=Huber, orange=CLEF

    for metric in METRICS:
        means  = [summary[mk][metric]["mean"]      for mk in models]
        errors = [summary[mk][metric]["ci_margin"] for mk in models]

        fig, ax = plt.subplots(figsize=(5, 4))
        x       = np.arange(len(models))
        bars    = ax.bar(x, means, yerr=errors, capsize=7,
                         color=colors[:len(models)], edgecolor="black",
                         linewidth=0.7, width=0.45)

        # Annotate bars with mean values
        for bar, mean, err in zip(bars, means, errors):
            if not np.isnan(mean):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        mean + err + 0.001,
                        f"{mean:.4f}", ha="center", va="bottom", fontsize=8)

        # Highlight the better model
        lower_better = LOWER_BETTER[metric]
        valid = [(i, m) for i, m in enumerate(means) if not np.isnan(m)]
        if valid:
            best_i = min(valid, key=lambda t: t[1] if lower_better else -t[1])[0]
            bars[best_i].set_edgecolor("#2d7d46")
            bars[best_i].set_linewidth(2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
        ax.set_title(f"{METRIC_LABELS[metric]}", fontsize=10)
        ax.grid(True, axis="y", alpha=0.35)
        plt.tight_layout()

        save_path = os.path.join(fig_dir, f"cmp_{metric}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {save_path}")

    # Combined 2×3 panel figure
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, metric in zip(axes.flat, METRICS):
        means  = [summary[mk][metric]["mean"]      for mk in models]
        errors = [summary[mk][metric]["ci_margin"] for mk in models]
        x      = np.arange(len(models))
        ax.bar(x, means, yerr=errors, capsize=5, color=colors[:len(models)],
               edgecolor="black", linewidth=0.6, width=0.45)
        ax.set_xticks(x)
        short = ["RHN\n(MSE)", "RHN\n(Huber)", "Ours\n(+CLEF)"]
        ax.set_xticklabels(short[:len(models)], fontsize=8)
        ax.set_title(METRIC_LABELS[metric], fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ReHeartNet (MSE) vs. + Huber vs. + CLEF  (8-fold CV, BIDMC)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    panel_path = os.path.join(fig_dir, "comparison_panel.png")
    plt.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved panel → {panel_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Reconstruction overlay: one figure showing both models on the same window
# ──────────────────────────────────────────────────────────────────────────────

def _save_recon_figure(model, test_loader, test_subs, device,
                       fold_idx, model_key, fig_dir) -> None:
    model.eval()
    with torch.no_grad():
        ppg_b, ecg_b = next(iter(test_loader))
        pred_b = model(ppg_b.to(device)).cpu()

    true_np = ecg_b.squeeze(-1).numpy()
    pred_np = pred_b.squeeze(-1).numpy()

    plot_reconstruction_samples(
        true_np, pred_np,
        subject_ids=[test_subs[min(i, len(test_subs)-1)] for i in range(min(4, len(true_np)))],
        fold_idx=fold_idx,
        save_path=os.path.join(fig_dir, f"recon_{model_key}_fold{fold_idx:02d}.png"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    device = config.DEVICE
    print(f"Device: {device}")

    # Load best Optuna params (shared hyperparameters for both models)
    best_params_path = os.path.join("results", "best_hyperparams.json")
    best_params: dict = {}
    if os.path.exists(best_params_path):
        with open(best_params_path) as f:
            best_params = json.load(f)
        print(f"Loaded hyperparameters from {best_params_path}")
    else:
        print("No best_hyperparams.json — using config defaults.")

    # Build shared frozen encoders (once for the whole run)
    print(f"Loading CLEF encoder ({args.clef_size}) ...")
    clef_encoder = load_clef_encoder(args.clef_path, args.clef_size, device)
    ptbxl_clf    = get_ptbxl_classifier(args.classifier_path).to(device)

    # CV splits (same random seed → identical folds for both models)
    all_subjects = get_all_record_names()
    n_folds = 2 if args.dry_run else args.n_folds
    splits  = get_cv_splits(
        all_subjects, n_splits=n_folds,
        save_path=os.path.join(args.output_dir, "fold_assignments.json"),
    )

    # ── Run both models ──────────────────────────────────────────────────────
    all_results = {}
    for model_key, model_cfg in MODELS.items():
        print(f"\n{'━'*60}")
        print(f"  Model : {model_cfg['label']}")
        print(f"{'━'*60}")
        all_results[model_key] = _run_model(
            model_key, model_cfg, args, device,
            clef_encoder, ptbxl_clf, splits, best_params,
        )

    # ── Comparison output ────────────────────────────────────────────────────
    print("\n" + "━"*60)
    print("  Building comparison outputs")
    print("━"*60)
    summary = _save_comparison_json(all_results, args.output_dir)
    _print_comparison_table(summary)
    _plot_side_by_side(summary, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")
    print("Key files:")
    print(f"  {args.output_dir}/comparison.json       ← side-by-side metrics")
    print(f"  {args.output_dir}/figures/comparison_panel.png  ← 2×3 panel figure")
    print(f"  {args.output_dir}/figures/cmp_*.png     ← per-metric bar charts")


if __name__ == "__main__":
    main()
