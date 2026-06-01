"""Clinical evaluation metrics for ECG reconstruction quality.

All functions accept numpy arrays of shape (N_windows, seq_len) where each
row is one 10-second ECG window.  R-peak-based metrics skip windows that
yield fewer than 3 detected peaks to avoid artifact contamination.
"""

import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ks_2samp, wasserstein_distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_rr_intervals(
    ecg_windows: np.ndarray,
    fs: int = 125,
    min_peaks: int = 3,
) -> np.ndarray:
    """Detect R-peaks in every window and return all RR intervals in seconds.

    Args:
        ecg_windows: (N, seq_len) array of ECG windows.
        fs:          Sampling frequency in Hz.
        min_peaks:   Windows with fewer detected peaks are skipped.

    Returns:
        1-D array of RR intervals in seconds (may be empty if no valid windows).
    """
    try:
        import neurokit2 as nk  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("neurokit2 is required for R-peak detection.") from e

    rr_all = []
    for window in ecg_windows:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, info = nk.ecg_peaks(window.astype(float), sampling_rate=fs, method="pantompkins1985")
                peaks = info["ECG_R_Peaks"]
            except Exception:
                continue
        if len(peaks) < min_peaks:
            continue
        rr_samples = np.diff(peaks)
        rr_all.extend(rr_samples / fs)

    return np.array(rr_all, dtype=np.float64)


def compute_confidence_interval(
    values: List[float],
) -> Tuple[float, float, float]:
    """Return (mean, ci95_low, ci95_high) using the standard 95% CI formula.

    CI = mean ± 1.96 * std / sqrt(N).  Returns (nan, nan, nan) for empty input.
    """
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return mean, mean, mean
    margin = 1.96 * float(np.std(arr, ddof=1)) / np.sqrt(len(arr))
    return mean, mean - margin, mean + margin


# ---------------------------------------------------------------------------
# Morphological metric
# ---------------------------------------------------------------------------

def compute_prd(true: np.ndarray, pred: np.ndarray) -> float:
    """Percentage Root Mean Square Difference (PRD).

    PRD = sqrt(sum((true - pred)^2) / sum(true^2)) * 100

    Lower is better.  Operates on flattened arrays.
    """
    true_flat = true.ravel().astype(np.float64)
    pred_flat = pred.ravel().astype(np.float64)
    denom = np.sum(true_flat ** 2)
    if denom < 1e-12:
        return float("nan")
    return float(np.sqrt(np.sum((true_flat - pred_flat) ** 2) / denom) * 100.0)


# ---------------------------------------------------------------------------
# Diagnostic metric
# ---------------------------------------------------------------------------

def compute_bce(
    true_ecg: np.ndarray,
    pred_ecg: np.ndarray,
    classifier: nn.Module,
    device: torch.device,
    batch_size: int = 64,
    fs: int = 125,
) -> float:
    """Binary Cross-Entropy between pathology probability vectors.

    Passes both real and reconstructed ECG windows through a frozen classifier,
    then computes BCE(P_real, P_recon) averaged over all windows and classes.

    BCE = -(1/C) * sum[ P_real * log(P_recon) + (1-P_real) * log(1-P_recon) ]

    Lower is better (closer to zero means reconstruction preserves diagnostics).

    Args:
        true_ecg:   (N, seq_len) ground-truth ECG windows.
        pred_ecg:   (N, seq_len) reconstructed ECG windows.
        classifier: Frozen nn.Module: (B, 1, seq_len) → (B, C) sigmoid.
        device:     Compute device.
        batch_size: Mini-batch size for classifier inference.
        fs:         Sampling frequency (for future normalisation hooks).
    """
    classifier.eval()
    eps = 1e-7
    bce_vals = []

    N = len(true_ecg)
    for start in range(0, N, batch_size):
        t_batch = true_ecg[start : start + batch_size]
        p_batch = pred_ecg[start : start + batch_size]

        t_tensor = torch.from_numpy(t_batch.astype(np.float32)).unsqueeze(1).to(device)
        p_tensor = torch.from_numpy(p_batch.astype(np.float32)).unsqueeze(1).to(device)

        with torch.no_grad():
            p_real  = classifier(t_tensor).cpu().numpy()  # (B, C)
            p_recon = classifier(p_tensor).cpu().numpy()  # (B, C)

        p_recon = np.clip(p_recon, eps, 1.0 - eps)
        bce = -(
            p_real * np.log(p_recon)
            + (1.0 - p_real) * np.log(1.0 - p_recon)
        ).mean(axis=1)                                    # (B,)
        bce_vals.extend(bce.tolist())

    return float(np.mean(bce_vals)) if bce_vals else float("nan")


# ---------------------------------------------------------------------------
# Rhythm metrics
# ---------------------------------------------------------------------------

def compute_emd(
    true_ecg: np.ndarray,
    pred_ecg: np.ndarray,
    fs: int = 125,
) -> float:
    """Earth Mover's Distance (Wasserstein-1) between RR interval distributions.

    Lower is better (closer to 0 means rhythm preserved).
    Returns nan if too few R-peaks are detected in either signal.
    """
    rr_true = extract_rr_intervals(true_ecg, fs=fs)
    rr_pred = extract_rr_intervals(pred_ecg, fs=fs)
    if len(rr_true) < 2 or len(rr_pred) < 2:
        return float("nan")
    return float(wasserstein_distance(rr_true, rr_pred))


def compute_ks(
    true_ecg: np.ndarray,
    pred_ecg: np.ndarray,
    fs: int = 125,
) -> Tuple[float, float]:
    """Kolmogorov-Smirnov test between RR interval CDFs.

    Returns:
        (D_statistic, p_value).
        D closer to 0 and p_value > 0.05 indicate statistically similar rhythms.
    """
    rr_true = extract_rr_intervals(true_ecg, fs=fs)
    rr_pred = extract_rr_intervals(pred_ecg, fs=fs)
    if len(rr_true) < 2 or len(rr_pred) < 2:
        return float("nan"), float("nan")
    stat, pval = ks_2samp(rr_true, rr_pred)
    return float(stat), float(pval)


def compute_beat_timing_mae(
    true_ecg: np.ndarray,
    pred_ecg: np.ndarray,
    fs: int = 125,
    min_peaks: int = 3,
) -> float:
    """Mean Absolute Error of R-peak times in seconds.

    Each detected peak in the reconstructed signal is matched to the
    nearest ground-truth peak (nearest-neighbour, no tolerance window).
    Returns nan if too few peaks are found in either signal.
    """
    try:
        import neurokit2 as nk  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("neurokit2 is required for R-peak detection.") from e

    errors = []
    for true_w, pred_w in zip(true_ecg, pred_ecg):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, info_t = nk.ecg_peaks(true_w.astype(float), sampling_rate=fs, method="pantompkins1985")
                _, info_p = nk.ecg_peaks(pred_w.astype(float), sampling_rate=fs, method="pantompkins1985")
            except Exception:
                continue
        peaks_t = np.array(info_t["ECG_R_Peaks"])
        peaks_p = np.array(info_p["ECG_R_Peaks"])
        if len(peaks_t) < min_peaks or len(peaks_p) < min_peaks:
            continue
        # Match each predicted peak to nearest ground-truth peak
        for p in peaks_p:
            nearest = peaks_t[np.argmin(np.abs(peaks_t - p))]
            errors.append(abs(p - nearest) / fs)

    return float(np.mean(errors)) if errors else float("nan")
