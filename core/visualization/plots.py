"""Visualization utilities for ReHeartNet CV results.

All functions save figures under results/figures/ (auto-created).
"""

import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

_FIG_DIR = os.path.join("results", "figures")


def _ensure_fig_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)


# ---------------------------------------------------------------------------
# Per-fold figures
# ---------------------------------------------------------------------------

def plot_reconstruction_samples(
    true_ecg: np.ndarray,
    pred_ecg: np.ndarray,
    fs: int = 125,
    subject_ids: Optional[List[str]] = None,
    n_samples: int = 4,
    save_path: Optional[str] = None,
    fold_idx: Optional[int] = None,
) -> None:
    """Overlay real vs reconstructed ECG for up to n_samples windows.

    Args:
        true_ecg:    (N, seq_len) ground-truth windows.
        pred_ecg:    (N, seq_len) reconstructed windows.
        fs:          Sampling frequency.
        subject_ids: Optional labels per window for title.
        n_samples:   Number of subplot rows to show.
        save_path:   Full path for the saved PNG. Auto-generated if None.
        fold_idx:    Fold index used in auto-generated filename.
    """
    if save_path is None:
        tag = f"fold{fold_idx:02d}" if fold_idx is not None else "preview"
        save_path = os.path.join(_FIG_DIR, f"reconstruction_{tag}.png")
    _ensure_fig_dir(save_path)

    n = min(n_samples, len(true_ecg))
    t = np.arange(true_ecg.shape[1]) / fs

    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), squeeze=False)
    for i in range(n):
        ax = axes[i, 0]
        ax.plot(t, true_ecg[i], color="#1f77b4", linewidth=1.0, label="Ground truth", alpha=0.9)
        ax.plot(t, pred_ecg[i], color="#ff7f0e", linewidth=1.0, label="Reconstructed", alpha=0.85)
        title = f"Sample {i}"
        if subject_ids is not None and i < len(subject_ids):
            title += f"  [{subject_ids[i]}]"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Amplitude (z)", fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("ECG Reconstruction: Ground Truth vs Predicted", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rr_distributions(
    true_rr: np.ndarray,
    pred_rr: np.ndarray,
    subject_id: str = "",
    fold_idx: Optional[int] = None,
    save_path: Optional[str] = None,
) -> None:
    """Overlaid KDE plots of RR interval distributions (real vs reconstructed).

    Args:
        true_rr:    1-D array of RR intervals (seconds) from ground-truth ECG.
        pred_rr:    1-D array of RR intervals (seconds) from reconstructed ECG.
        subject_id: Optional label for the figure title.
        fold_idx:   Used in auto-generated filename.
        save_path:  Full path for the saved PNG. Auto-generated if None.
    """
    if save_path is None:
        tag = f"fold{fold_idx:02d}" if fold_idx is not None else "preview"
        save_path = os.path.join(_FIG_DIR, f"rr_dist_{tag}.png")
    _ensure_fig_dir(save_path)

    fig, ax = plt.subplots(figsize=(8, 4))

    for rr, label, color in [
        (true_rr, "Ground truth", "#1f77b4"),
        (pred_rr, "Reconstructed", "#ff7f0e"),
    ]:
        if len(rr) < 2:
            continue
        kde = gaussian_kde(rr, bw_method="scott")
        x_grid = np.linspace(max(0.3, rr.min() - 0.1), min(2.0, rr.max() + 0.1), 300)
        ax.fill_between(x_grid, kde(x_grid), alpha=0.35, color=color)
        ax.plot(x_grid, kde(x_grid), color=color, linewidth=1.5, label=label)

    title = "RR Interval Distribution"
    if subject_id:
        title += f"  [{subject_id}]"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("RR interval (s)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    fold_idx: int,
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation loss curves for a single fold.

    Args:
        train_losses: Per-epoch training losses.
        val_losses:   Per-epoch validation losses.
        fold_idx:     Fold index for title and filename.
        save_path:    Full path for the saved PNG. Auto-generated if None.
    """
    if save_path is None:
        save_path = os.path.join(_FIG_DIR, f"loss_curves_fold{fold_idx:02d}.png")
    _ensure_fig_dir(save_path)

    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="Train loss", color="#1f77b4", linewidth=1.2)
    ax.plot(epochs, val_losses,   label="Val loss",   color="#d62728", linewidth=1.2)
    ax.set_title(f"Loss Curves — Fold {fold_idx:02d}", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Post-CV aggregate figures
# ---------------------------------------------------------------------------

def plot_fold_metrics_bar(
    fold_metrics: List[Dict],
    metric_name: str,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Bar chart of a metric across all folds with 95% CI error bar.

    Args:
        fold_metrics: List of per-fold metric dicts (must contain metric_name).
        metric_name:  Key to extract from each dict.
        save_path:    Full path for the saved PNG.
        title:        Optional figure title override.
    """
    if save_path is None:
        save_path = os.path.join(_FIG_DIR, f"metric_bar_{metric_name}.png")
    _ensure_fig_dir(save_path)

    values = [m[metric_name] for m in fold_metrics if not np.isnan(m.get(metric_name, float("nan")))]
    if not values:
        return

    fold_labels = [str(m.get("fold", i)) for i, m in enumerate(fold_metrics)]
    mean_val = np.mean(values)
    ci_margin = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(max(6, len(fold_metrics) * 0.6), 4))
    raw_vals = [m.get(metric_name, float("nan")) for m in fold_metrics]
    colors = ["#1f77b4" if not np.isnan(v) else "#cccccc" for v in raw_vals]
    ax.bar(fold_labels, [v if not np.isnan(v) else 0 for v in raw_vals], color=colors, alpha=0.8)
    ax.axhline(mean_val, color="red", linestyle="--", linewidth=1.2, label=f"Mean={mean_val:.3f}")
    ax.axhline(mean_val + ci_margin, color="red", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axhline(mean_val - ci_margin, color="red", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.fill_between(range(len(fold_labels)), mean_val - ci_margin, mean_val + ci_margin,
                    alpha=0.12, color="red", label=f"±95% CI ({ci_margin:.3f})")
    ax.set_title(title or f"{metric_name} per fold", fontsize=10)
    ax.set_xlabel("Fold", fontsize=9)
    ax.set_ylabel(metric_name, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_boxplots(
    fold_metrics: List[Dict],
    metric_names: Optional[List[str]] = None,
    save_path: str = os.path.join(_FIG_DIR, "metric_boxplots.png"),
) -> None:
    """Box plots for all evaluation metrics across folds.

    Args:
        fold_metrics:  List of per-fold metric dicts.
        metric_names:  Metrics to plot. Defaults to the standard five.
        save_path:     Full path for the saved PNG.
    """
    _ensure_fig_dir(save_path)

    if metric_names is None:
        metric_names = ["prd", "bce", "emd", "ks_stat", "pearson_r", "beat_timing_mae"]

    data = []
    labels = []
    for m in metric_names:
        vals = [fm[m] for fm in fold_metrics if not np.isnan(fm.get(m, float("nan")))]
        if vals:
            data.append(vals)
            labels.append(m)

    if not data:
        return

    fig, axes = plt.subplots(1, len(data), figsize=(3 * len(data), 5), squeeze=False)
    for ax, vals, lbl in zip(axes[0], data, labels):
        bp = ax.boxplot(vals, patch_artist=True, widths=0.6)
        for patch in bp["boxes"]:
            patch.set_facecolor("#aec6e8")
        ax.set_title(lbl, fontsize=9)
        ax.set_ylabel(lbl, fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Evaluation Metrics Across CV Folds", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

def save_results_summary(
    fold_metrics: List[Dict],
    metric_names: Optional[List[str]] = None,
    save_path: str = os.path.join("results", "summary.json"),
) -> None:
    """Compute mean ± 95% CI for each metric and save to JSON.

    Args:
        fold_metrics:  List of per-fold metric dicts.
        metric_names:  Metrics to summarise. Defaults to the standard set.
        save_path:     Output path for the JSON file.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    if metric_names is None:
        metric_names = ["prd", "pearson_r", "bce", "emd", "ks_stat", "ks_pvalue", "beat_timing_mae"]

    summary = {}
    for m in metric_names:
        vals = [fm[m] for fm in fold_metrics if not np.isnan(fm.get(m, float("nan")))]
        if not vals:
            summary[m] = {"mean": None, "std": None, "ci95_low": None, "ci95_high": None, "per_fold": []}
            continue
        arr = np.array(vals)
        mean   = float(np.mean(arr))
        std    = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        margin = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        summary[m] = {
            "mean":      mean,
            "std":       std,
            "ci95_low":  mean - margin,
            "ci95_high": mean + margin,
            "per_fold":  vals,
        }

    # Include raw fold records for traceability
    summary["fold_records"] = fold_metrics

    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Results summary saved to {save_path}")
    print("\n--- Summary (mean ± 95% CI) ---")
    for m, s in summary.items():
        if m == "fold_records":
            continue
        if s["mean"] is not None:
            print(f"  {m:20s}: {s['mean']:.4f}  [{s['ci95_low']:.4f}, {s['ci95_high']:.4f}]")
