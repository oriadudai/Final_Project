"""Three-way comparison: original ReHeartNet vs. our loss-function improvements.

Three model variants, all using the same DC-BiLSTM architecture and the same
training protocol as Lee et al. (2026) — batch=1, lr=1e-2, ×0.75 linear decay
every 50 epochs, 1000 epochs, no early stopping.  Only the loss function changes:

  1. reheartnet_original -- faithful replica of Lee et al.:
       MSE loss, 4 s windows, FIR bandpass, paper training protocol.

  2. reheartnet_huber    -- same protocol, Huber loss.
       Optuna tunes huber_delta and hidden_size.

  3. reheartnet_clef     -- same protocol, Huber + CLEF composite loss.
       10 s windows (CLEF encoder requires 10 s input).
       Optuna tunes lambda_clinical, huber_delta, and hidden_size.

Outputs:
  results/comparison_reheartnet/summary_{model}.json   per-model summary
  results/comparison_reheartnet/comparison.json        side-by-side table
  results/comparison_reheartnet/figures/               all plots

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
from core.data_loader import build_group_fold, build_test_dataset, get_cv_splits, split_train_val
from core.metrics.clinical_metrics import compute_bce
from core.models.ptbxl_classifier import build_classifier
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

# ------------------------------------------------------------------------------
# Three model variants -- same DC-BiLSTM architecture, same training protocol
# as Lee et al. (2026), differing only in loss function.
#
#  original : MSE                   <- paper baseline
#  huber    : Huber                 <- our contribution 1
#  clef     : Huber + CLEF          <- our full method
#
# All variants:
#   - 8-fold subject-grouped CV (same splits, same random seed)
#   - Training protocol from Lee et al.: batch=1, lr=1e-2, ×0.75/50ep, 1000ep
#   - Unspecified paper values (hidden_size, loss HPs) -> Optuna / config default
#
# Keys in each entry:
#   loss_type          : "mse" | "huber" | "clef"
#   window_sec         : float (seconds) or None -> config default (10 s)
#   overlap_frac       : 0.0 = non-overlapping, 0.5 = 50% overlap
#   apply_bandpass     : True -> FIR ECG 0.5-55 Hz / PPG 0.5-10 Hz (Lee et al.)
#   batch_size / epochs / lr / lr_schedule / early_stop_patience: training protocol
#   use_optuna         : bool -- if False, best_params never consulted;
#                        unspecified values fall back to config defaults
# ------------------------------------------------------------------------------
MODELS = {
    "reheartnet_original": {
        "label":               "ReHeartNet (original)",
        "loss_type":           "mse",
        # preprocessing -- matches Lee et al. supplementary exactly
        "window_sec":          4.0,
        "overlap_frac":        0.0,
        "apply_bandpass":      True,
        # training -- matches Lee et al. supplementary exactly
        "batch_size":          1,
        "epochs":              1000,
        "lr":                  1e-2,
        "lr_schedule":         "linear_decay",
        "early_stop_patience": None,          # run all 1000 epochs
        # Paper did not use Optuna; anything not above falls back to config defaults
        "use_optuna":          False,
    },
    "reheartnet_huber": {
        "label":               "ReHeartNet + Huber (ours)",
        "loss_type":           "huber",
        # Same preprocessing and training protocol as the original paper.
        # Only the loss function changes (MSE -> Huber).
        # Optuna tunes loss hyperparameters (huber_delta) and hidden_size.
        "window_sec":          4.0,
        "overlap_frac":        0.0,
        "apply_bandpass":      True,
        "batch_size":          1,
        "epochs":              1000,
        "lr":                  1e-2,
        "lr_schedule":         "linear_decay",
        "early_stop_patience": None,
    },
    "reheartnet_clef": {
        "label":               "ReHeartNet + Huber + CLEF (ours)",
        "loss_type":           "clef",
        # 10 s window required: CLEF encoder expects 10 s input (5000 samples at
        # 500 Hz). Feeding it 4 s would distort temporal content and corrupt BCE.
        # All other training settings match the paper.
        # Optuna tunes loss hyperparameters (lambda_clinical, huber_delta) and hidden_size.
        "window_sec":          None,           # config default = 10 s = 1250 samples
        "overlap_frac":        0.5,
        "apply_bandpass":      False,
        "batch_size":          1,
        "epochs":              1000,
        "lr":                  1e-2,
        "lr_schedule":         "linear_decay",
        "early_stop_patience": None,
    },
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


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------
# Single-model CV loop
# ------------------------------------------------------------------------------

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
    """Run CV for one model variant. Returns list of fold metric dicts.

    Hyperparameter priority (highest -> lowest):
      model_cfg explicit value  >  args CLI override  >  best_params (Optuna)  >  config default

    For reheartnet_original, use_optuna=False so best_params is never consulted —
    only paper-specified values (in model_cfg) or config defaults are used.
    """
    # The original paper fixed hyperparameters manually; never pull Optuna values for it.
    hp = best_params if model_cfg.get("use_optuna", True) else {}

    epochs      = model_cfg.get("epochs")     or args.epochs     or int(hp.get("epochs",           config.EPOCHS))
    batch_size  = model_cfg.get("batch_size") or args.batch_size or int(hp.get("batch_size",       config.BATCH_SIZE))
    hidden_size =                                                    int(hp.get("hidden_size",      config.HIDDEN_SIZE))
    lr          = model_cfg.get("lr")         or                  float(hp.get("lr",               config.LEARNING_RATE))
    huber_delta =                                                  float(hp.get("huber_delta",      config.HUBER_DELTA))
    lambda_clinical =                                              float(hp.get("lambda_clinical",  config.LAMBDA_CLINICAL))

    loss_type           = model_cfg["loss_type"]
    window_sec          = model_cfg.get("window_sec")           # None -> config default
    overlap_frac        = model_cfg.get("overlap_frac", 0.5)
    apply_bandpass      = model_cfg.get("apply_bandpass", False)
    lr_schedule         = model_cfg.get("lr_schedule", "plateau")
    early_stop_patience = model_cfg.get("early_stop_patience", 15)

    if args.dry_run:
        epochs = 2
        # Dry-run overrides: pipeline correctness check, not faithful hyperparameters.
        batch_size = max(batch_size, 32)   # batch=1 makes CPU dry-runs impractically slow

    label = model_cfg["label"]
    n_folds = len(splits)
    print(f"\n  loss={loss_type}  |  lr={lr}  |  H={hidden_size}  |  epochs={epochs}  |  folds={n_folds}")
    if window_sec is not None:
        print(f"  window={window_sec}s  overlap={overlap_frac:.0%}  bandpass={apply_bandpass}")

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

        # Limit subjects per fold in dry-run so training completes in seconds on CPU
        _train = train_subs[:3] if args.dry_run else train_subs
        _test  = test_subs[:1]  if args.dry_run else test_subs
        train_ds, test_ds = build_group_fold(
            _train, _test, apply_align=True,
            window_sec=window_sec, overlap_frac=overlap_frac,
            apply_bandpass=apply_bandpass,
        )
        train_inner, val_ds = split_train_val(train_ds, val_fraction=0.1)

        train_loader = DataLoader(train_inner, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,      batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        test_loader  = DataLoader(test_ds,     batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        model, history = train_fold(
            fold_idx            = fold_idx,
            fold_subjects       = test_subs,
            train_loader        = train_loader,
            val_loader          = val_loader,
            clef_encoder        = clef_encoder,
            device              = device,
            epochs              = epochs,
            lr                  = lr,
            lambda_clinical     = lambda_clinical,
            huber_delta         = huber_delta,
            hidden_size         = hidden_size,
            checkpoint_dir      = os.path.join(args.output_dir, "checkpoints"),
            use_wandb           = not args.no_wandb,
            model_name          = "reheartnet",   # same DC-BiLSTM architecture for all variants
            loss_type           = loss_type,
            lr_schedule         = lr_schedule,
            early_stop_patience = early_stop_patience,
        )

        if window_sec is not None:
            # 4 s model: evaluate all metrics except BCE on the 4 s test windows,
            # then compute BCE separately on 10 s windows (CLEFClassifier needs 10 s input).
            # The BiLSTM is sequence-length agnostic so the trained model runs on 10 s fine.
            metrics = evaluate_fold(model, test_loader, ptbxl_clf, device, clef_encoder=None)

            bce_ds = build_test_dataset(
                _test, window_sec=None, overlap_frac=0.5, apply_bandpass=False,
            )
            bce_loader = DataLoader(bce_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=0, pin_memory=True)
            model.eval()
            true_10s, pred_10s = [], []
            with torch.no_grad():
                for ppg_b, ecg_b in bce_loader:
                    pred_b = model(ppg_b.to(device))
                    true_10s.append(ecg_b.squeeze(-1).cpu().numpy())
                    pred_10s.append(pred_b.squeeze(-1).cpu().numpy())
            true_10s = np.concatenate(true_10s)
            pred_10s = np.concatenate(pred_10s)
            clf = build_classifier(clef_encoder).to(device)
            metrics["bce"] = compute_bce(true_10s, pred_10s, clf, device)
        else:
            # CLEF model: already uses 10 s windows — standard evaluation path.
            metrics = evaluate_fold(model, test_loader, ptbxl_clf, device, clef_encoder=clef_encoder)

        metrics.update({"fold": fold_idx, "subjects": test_subs, "model": model_key})
        fold_metrics.append(metrics)

        print(
            f"    [{label}] fold {fold_idx:02d} -- "
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


# ------------------------------------------------------------------------------
# Comparison output
# ------------------------------------------------------------------------------

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
    print(f"\nComparison saved -> {path}")
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
    """One figure per metric: side-by-side bars with CI for all model variants."""
    fig_dir = os.path.join(output_dir, "figures")
    models  = list(summary.keys())
    labels  = [summary[mk]["label"] for mk in models]
    # original=grey, Huber=green, CLEF=orange
    colors  = ["#b0b0b0", "#a8d5a2", "#e8906a"]

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
        short = ["RHN\n(orig)", "RHN\n+Huber", "RHN\n+CLEF"]
        ax.set_xticklabels(short[:len(models)], fontsize=8)
        ax.set_title(METRIC_LABELS[metric], fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ReHeartNet: MSE vs. +Huber vs. +Huber+CLEF  (8-fold CV, BIDMC)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    panel_path = os.path.join(fig_dir, "comparison_panel.png")
    plt.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved panel -> {panel_path}")


# ------------------------------------------------------------------------------
# Reconstruction overlay: one figure showing both models on the same window
# ------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    device = config.DEVICE
    print(f"Device: {device}")

    # Load best Optuna params (used as defaults for our three variants)
    best_params_path = os.path.join("results", "best_hyperparams.json")
    best_params: dict = {}
    if os.path.exists(best_params_path):
        with open(best_params_path) as f:
            best_params = json.load(f)
        print(f"Loaded hyperparameters from {best_params_path}")
    else:
        print("No best_hyperparams.json -- using config defaults.")

    # Build shared frozen encoders (once for the whole run)
    print(f"Loading CLEF encoder ({args.clef_size}) ...")
    clef_encoder = load_clef_encoder(args.clef_path, args.clef_size, device)
    ptbxl_clf    = get_ptbxl_classifier(args.classifier_path).to(device)

    all_subjects = get_all_record_names()

    # All three variants share the same subject-grouped 8-fold CV splits
    # (subjects are fully separated between folds -- no subject leakage).
    n_folds = 2 if args.dry_run else args.n_folds
    splits = get_cv_splits(
        all_subjects, n_splits=n_folds,
        save_path=os.path.join(args.output_dir, "fold_assignments.json"),
    )

    # -- Run all three model variants -----------------------------------------
    all_results = {}
    for model_key, model_cfg in MODELS.items():
        print(f"\n{'-'*60}")
        print(f"  Model : {model_cfg['label']}")
        print(f"{'-'*60}")
        all_results[model_key] = _run_model(
            model_key, model_cfg, args, device,
            clef_encoder, ptbxl_clf, splits, best_params,
        )

    # -- Comparison output ----------------------------------------------------
    print("\n" + "-"*60)
    print("  Building comparison outputs")
    print("-"*60)
    summary = _save_comparison_json(all_results, args.output_dir)
    _print_comparison_table(summary)
    _plot_side_by_side(summary, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")
    print("Key files:")
    print(f"  {args.output_dir}/comparison.json                  <- side-by-side metrics")
    print(f"  {args.output_dir}/figures/comparison_panel.png     <- 2×3 panel figure")
    print(f"  {args.output_dir}/figures/cmp_*.png                <- per-metric bar charts")
    print(f"\nAll models evaluated with {n_folds}-fold subject-grouped CV ({len(all_subjects)} subjects).")


if __name__ == "__main__":
    main()
