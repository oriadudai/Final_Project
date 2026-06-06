"""Evaluation pipeline for ReHeartNet CV folds.

evaluate_fold() runs full clinical evaluation on a held-out test DataLoader
and returns all four eval_logic metrics plus Pearson r.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import core.config as config
from core.metrics.clinical_metrics import (
    compute_rmse,
    compute_prd,
    compute_bce,
    compute_emd,
    compute_ks,
    compute_beat_timing_mae,
)
from core.models.ptbxl_classifier import build_classifier


def compute_pearson_r(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Pearson correlation coefficient between two tensors (flattened)."""
    p = pred.detach().cpu().numpy().ravel()
    t = target.detach().cpu().numpy().ravel()
    corr = np.corrcoef(p, t)
    return float(corr[0, 1])


def _collect_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple:
    """Run inference and return stacked numpy arrays (N, seq_len)."""
    all_true, all_pred = [], []
    model.eval()
    with torch.no_grad():
        for ppg, ecg in test_loader:
            ppg = ppg.to(device)
            pred = model(ppg)
            all_true.append(ecg.squeeze(-1).cpu().numpy())   # (B, seq_len)
            all_pred.append(pred.squeeze(-1).cpu().numpy())  # (B, seq_len)
    true_arr = np.concatenate(all_true, axis=0)   # (N, seq_len)
    pred_arr = np.concatenate(all_pred, axis=0)   # (N, seq_len)
    return true_arr, pred_arr


def evaluate_fold(
    model: nn.Module,
    test_loader: DataLoader,
    ptbxl_classifier: nn.Module,
    device: torch.device,
    fs: int = config.FS,
    clef_encoder: nn.Module = None,
) -> Dict[str, float]:
    """Compute all clinical metrics for one CV fold test set.

    If clef_encoder is provided, BCE uses CLEFClassifier (pretrained 256-dim
    clinical features from CLEF backbone). Otherwise falls back to
    ptbxl_classifier (surrogate with random weights).

    Returns a dict with keys:
        prd             - Percentage Root Mean Square Difference (lower better)
        pearson_r       - Pearson correlation coefficient (higher better)
        bce             - BCE in CLEF clinical feature space (lower better)
        emd             - Earth Mover's Distance on RR intervals (lower better)
        ks_stat         - KS test D-statistic on RR intervals (lower better)
        ks_pvalue       - KS test p-value (>0.05 is desirable)
        beat_timing_mae - Mean absolute R-peak timing error in seconds (lower better)
    """
    # Use CLEF-based classifier if available, else fall back to surrogate
    classifier = build_classifier(clef_encoder) if clef_encoder is not None else ptbxl_classifier
    classifier = classifier.to(device)

    true_arr, pred_arr = _collect_predictions(model, test_loader, device)

    # Pearson r (computed sample-wise then averaged)
    pearson_vals = []
    for t, p in zip(true_arr, pred_arr):
        if np.std(t) > 1e-8 and np.std(p) > 1e-8:
            pearson_vals.append(float(np.corrcoef(t, p)[0, 1]))
    pearson_r = float(np.mean(pearson_vals)) if pearson_vals else float("nan")

    rmse     = compute_rmse(true_arr, pred_arr)
    prd      = compute_prd(true_arr, pred_arr)
    bce      = compute_bce(true_arr, pred_arr, classifier, device)
    emd      = compute_emd(true_arr, pred_arr, fs=fs)
    ks_stat, ks_pval = compute_ks(true_arr, pred_arr, fs=fs)
    beat_mae = compute_beat_timing_mae(true_arr, pred_arr, fs=fs)

    return {
        "rmse":            rmse,
        "prd":             prd,
        "pearson_r":       pearson_r,
        "bce":             bce,
        "emd":             emd,
        "ks_stat":         ks_stat,
        "ks_pvalue":       ks_pval,
        "beat_timing_mae": beat_mae,
    }
