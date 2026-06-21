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


def plot_calibration_comparison(
    true_ecg: np.ndarray,
    baseline_pred: np.ndarray,
    calibrated_pred: np.ndarray,
    fs: int = 125,
    n_samples: int = 3,
    save_path: Optional[str] = None,
    subject_id: Optional[str] = None,
    window_indices: Optional[List[int]] = None,
) -> None:
    """Overlay real ECG vs baseline (population) vs calibrated (per-subject) reconstructions.

    Args:
        true_ecg:        (N, seq_len) ground-truth eval-slice windows.
        baseline_pred:   (N, seq_len) reconstructions before per-subject calibration.
        calibrated_pred: (N, seq_len) reconstructions after per-subject calibration.
        fs:              Sampling frequency.
        n_samples:       Number of windows to show (ignored if window_indices given);
                          chosen evenly spaced across the eval slice.
        save_path:       Full path for the saved PNG. Auto-generated if None.
        subject_id:       Optional subject label for the figure title/filename.
        window_indices:  Explicit eval-slice window indices to plot.
    """
    if save_path is None:
        tag = subject_id or "preview"
        save_path = os.path.join(_FIG_DIR, f"calibration_comparison_{tag}.png")
    _ensure_fig_dir(save_path)

    if window_indices is None:
        n = min(n_samples, len(true_ecg))
        window_indices = np.linspace(0, len(true_ecg) - 1, n).astype(int)
    t = np.arange(true_ecg.shape[1]) / fs

    fig, axes = plt.subplots(len(window_indices), 1, figsize=(12, 3 * len(window_indices)), squeeze=False)
    for row, idx in enumerate(window_indices):
        ax = axes[row, 0]
        ax.plot(t, true_ecg[idx], color="#1f77b4", linewidth=1.2, label="Real ECG", alpha=0.9)
        ax.plot(t, baseline_pred[idx], color="#ff7f0e", linewidth=1.0, label="Baseline (population)", alpha=0.85)
        ax.plot(t, calibrated_pred[idx], color="#2ca02c", linewidth=1.0, label="Calibrated (per-subject)", alpha=0.85)
        title = f"Eval window {idx}"
        if subject_id:
            title += f"  [{subject_id}]"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Amplitude (z)", fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("ECG Reconstruction: Real vs Baseline vs Calibrated", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_comparison_multi_subject(
    subjects: List[Dict],
    fs: int = 125,
    save_path: Optional[str] = None,
    crop_sec: Optional[float] = 4.0,
) -> None:
    """One panel per subject (stacked vertically): real ECG vs baseline vs calibrated.

    Args:
        subjects: list of dicts, one per subject, each with keys:
            "subject_id":        label for the panel title.
            "true_ecg":          (seq_len,) ground-truth window.
            "baseline_pred":     (seq_len,) reconstruction before calibration.
            "calibrated_pred":   (seq_len,) reconstruction after calibration.
            "baseline_metrics" / "calibrated_metrics": optional dicts with
                "prd" and "pearson_r" keys, shown in the panel title.
        fs:        Sampling frequency.
        save_path: Full path for the saved PNG. Defaults to
                   results/figures/calibration_comparison_multi.png.
        crop_sec:  If set, only show the first crop_sec seconds of each window
                   (the full window is often too long to read individual
                   beats). None shows the full window.
    """
    if save_path is None:
        save_path = os.path.join(_FIG_DIR, "calibration_comparison_multi.png")
    _ensure_fig_dir(save_path)

    n = len(subjects)
    seq_len = subjects[0]["true_ecg"].shape[0]
    n_crop = min(seq_len, int(round(crop_sec * fs))) if crop_sec else seq_len
    t = np.arange(n_crop) / fs

    fig, axes = plt.subplots(n, 1, figsize=(10, 3.6 * n), squeeze=False)
    for row, s in enumerate(subjects):
        ax = axes[row, 0]
        ax.plot(t, s["true_ecg"][:n_crop], color="#1f77b4", linewidth=1.4, label="Real ECG", alpha=0.9)
        ax.plot(t, s["baseline_pred"][:n_crop], color="#ff7f0e", linewidth=1.2, label="Baseline (population)", alpha=0.85)
        ax.plot(t, s["calibrated_pred"][:n_crop], color="#2ca02c", linewidth=1.2, label="Calibrated (per-subject)", alpha=0.85)
        title = s.get("subject_id", f"Subject {row}")
        bm, cm = s.get("baseline_metrics"), s.get("calibrated_metrics")
        if bm and cm:
            if "emd" in bm and "emd" in cm:
                title += (
                    f"\nBaseline:   RMSE={bm['rmse']:.3f}  PRD={bm['prd']:5.1f}%  r={bm['pearson_r']:+.2f}  "
                    f"EMD={bm['emd']:.3f}  KS={bm['ks_stat']:.3f}  beat-MAE={bm['beat_timing_mae']:.3f}s\n"
                    f"Calibrated: RMSE={cm['rmse']:.3f}  PRD={cm['prd']:5.1f}%  r={cm['pearson_r']:+.2f}  "
                    f"EMD={cm['emd']:.3f}  KS={cm['ks_stat']:.3f}  beat-MAE={cm['beat_timing_mae']:.3f}s"
                )
            else:
                title += (f"   PRD {bm['prd']:.1f}%→{cm['prd']:.1f}%, "
                          f"r {bm['pearson_r']:+.2f}→{cm['pearson_r']:+.2f}")
        ax.set_title(title, fontsize=9, family="monospace", loc="left")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Amplitude (z)", fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("ECG Reconstruction: Real vs Baseline vs Calibrated", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _format_true_probs(true_probs_mean: Optional[List[float]], superclasses: Optional[List[str]]) -> str:
    """'True dx: NORM=0.77 MI=0.13 ...', sorted descending, or '' if unavailable."""
    if not true_probs_mean or not superclasses:
        return ""
    pairs = sorted(zip(superclasses, true_probs_mean), key=lambda x: -x[1])
    return "True dx (real ECG): " + "  ".join(f"{c}={p:.2f}" for c, p in pairs)


def plot_calibration_asymmetry(
    subjects: List[Dict],
    fs: int = 125,
    save_path: Optional[str] = None,
    crop_sec: Optional[float] = 4.0,
    baseline_loss_label: Optional[str] = None,
    superclasses: Optional[List[str]] = None,
) -> None:
    """One panel per subject: real ECG vs baseline vs two calibration variants.

    Illustrates the perception-distortion calibration asymmetry (negative-sum
    vs. coopetitive regime) by overlaying a distortion-only calibration
    (e.g. MSE) against a composite calibration on the same baseline window.

    Args:
        subjects: list of dicts, one per subject, each with keys:
            "subject_id":      label for the panel title.
            "true_ecg":        (seq_len,) ground-truth window.
            "baseline_pred":   (seq_len,) reconstruction before calibration.
            "calib_a_pred":    (seq_len,) reconstruction after calibration A.
            "calib_a_label":   legend label for calibration A (e.g. "MSE").
            "calib_b_pred":    (seq_len,) reconstruction after calibration B.
            "calib_b_label":   legend label for calibration B (e.g. "Composite").
            "baseline_metrics" / "calib_a_metrics" / "calib_b_metrics": optional
                dicts with "prd", "pearson_r", "emd", "ks_stat" keys, shown in
                the panel title.
        fs:        Sampling frequency.
        save_path: Full path for the saved PNG. Defaults to
                   results/figures/calibration_asymmetry.png.
        crop_sec:  Only show the first crop_sec seconds of each window
                   (default 4s, to keep the panel readable; the full window
                   is 10s for CLEF's window configuration).
        baseline_loss_label: Name of the loss the baseline (uncalibrated)
                   model was trained with (e.g. "Huber + CLEF"), shown next
                   to the "Baseline" legend entry and panel title so the
                   reader knows what objective produced the population model
                   being calibrated.
    """
    if save_path is None:
        save_path = os.path.join(_FIG_DIR, "calibration_asymmetry.png")
    _ensure_fig_dir(save_path)

    n = len(subjects)
    seq_len = subjects[0]["true_ecg"].shape[0]
    n_crop = min(seq_len, int(round(crop_sec * fs))) if crop_sec else seq_len
    t = np.arange(n_crop) / fs

    baseline_tag = f" (trained: {baseline_loss_label})" if baseline_loss_label else ""

    fig, axes = plt.subplots(n, 1, figsize=(11, 4.2 * n), squeeze=False)
    for row, s in enumerate(subjects):
        ax = axes[row, 0]
        ax.plot(t, s["true_ecg"][:n_crop], color="#1f77b4", linewidth=1.5, label="Real ECG", alpha=0.9, zorder=4)
        ax.plot(t, s["baseline_pred"][:n_crop], color="#7f7f7f", linewidth=1.0,
                label=f"Baseline{baseline_tag}", alpha=0.7, zorder=1)
        ax.plot(t, s["calib_a_pred"][:n_crop], color="#d62728", linewidth=1.1,
                label=f"+{s.get('calib_a_label', 'Calib A')} (calibrated)", alpha=0.85, zorder=2)
        ax.plot(t, s["calib_b_pred"][:n_crop], color="#2ca02c", linewidth=1.1,
                label=f"+{s.get('calib_b_label', 'Calib B')} (calibrated)", alpha=0.85, zorder=3)

        title = s.get("subject_id", f"Subject {row}")
        bm = s.get("baseline_metrics")
        am = s.get("calib_a_metrics")
        cm = s.get("calib_b_metrics")

        def _row(label, mm):
            line = (f"{label}: RMSE={mm['rmse']:.3f}  PRD={mm['prd']:5.1f}%  r={mm['pearson_r']:+.2f}  "
                     f"EMD={mm['emd']:.3f}  KS={mm['ks_stat']:.3f}")
            if "diag_kl" in mm and "flip_rate" in mm:
                line += f"  KL={mm['diag_kl']:.3f}  flip={mm['flip_rate']:.2f}"
            return line

        if bm and am and cm:
            title += (
                "\n" + _row(f"Baseline{baseline_tag}", bm)
                + "\n" + _row(f"+{s.get('calib_a_label', 'A')}", am)
                + "\n" + _row(f"+{s.get('calib_b_label', 'B')}", cm)
            )
        true_dx = _format_true_probs(s.get("true_probs_mean"), superclasses)
        if true_dx:
            title += "\n" + true_dx
        ax.set_title(title, fontsize=8.5, family="monospace", loc="left")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Amplitude (z)", fontsize=8)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Perception-Distortion Calibration Asymmetry: Real vs Baseline vs Two Calibration Objectives",
                 fontsize=10.5, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


_CALIB_COLOR_CYCLE = ["#d62728", "#9467bd", "#2ca02c", "#e377c2", "#8c564b"]


def plot_calibration_multi_loss(
    subjects: List[Dict],
    fs: int = 125,
    save_path: Optional[str] = None,
    crop_sec: Optional[float] = 4.0,
    baseline_loss_label: Optional[str] = None,
    suptitle: Optional[str] = None,
    superclasses: Optional[List[str]] = None,
) -> None:
    """One panel per subject: real ECG vs baseline vs an arbitrary number of
    calibration objectives (generalizes plot_calibration_asymmetry from 2 to N).

    Args:
        subjects: list of dicts, one per subject, each with keys:
            "subject_id":      label for the panel title.
            "true_ecg":        (seq_len,) ground-truth window.
            "baseline_pred":   (seq_len,) reconstruction before calibration.
            "baseline_metrics": optional dict with "rmse"/"prd"/"pearson_r"/
                "emd"/"ks_stat" (and optionally "diag_kl"/"flip_rate") keys.
            "calibrations":    list of dicts, one per calibration objective,
                each with "label" (e.g. "MSE"), "pred" (seq_len,), and
                optional "metrics" (same keys as baseline_metrics).
        fs:        Sampling frequency.
        save_path: Full path for the saved PNG. Defaults to
                   results/figures/calibration_multi_loss.png.
        crop_sec:  Only show the first crop_sec seconds of each window
                   (default 4s).
        baseline_loss_label: Name of the loss the baseline model was trained
                   with (e.g. "Huber + CLEF"), shown next to "Baseline".
        suptitle:  Figure-level title. Defaults to a generic description.
    """
    if save_path is None:
        save_path = os.path.join(_FIG_DIR, "calibration_multi_loss.png")
    _ensure_fig_dir(save_path)

    n = len(subjects)
    seq_len = subjects[0]["true_ecg"].shape[0]
    n_crop = min(seq_len, int(round(crop_sec * fs))) if crop_sec else seq_len
    t = np.arange(n_crop) / fs

    baseline_tag = f" (trained: {baseline_loss_label})" if baseline_loss_label else ""

    def _row(label, mm):
        line = (f"{label}: RMSE={mm['rmse']:.3f}  PRD={mm['prd']:5.1f}%  r={mm['pearson_r']:+.2f}  "
                 f"EMD={mm['emd']:.3f}  KS={mm['ks_stat']:.3f}")
        if "diag_kl" in mm and "flip_rate" in mm:
            line += f"  KL={mm['diag_kl']:.3f}  flip={mm['flip_rate']:.2f}"
        return line

    fig, axes = plt.subplots(n, 1, figsize=(11, 4.6 * n), squeeze=False)
    for row, s in enumerate(subjects):
        ax = axes[row, 0]
        ax.plot(t, s["true_ecg"][:n_crop], color="#1f77b4", linewidth=1.5, label="Real ECG", alpha=0.9, zorder=10)
        ax.plot(t, s["baseline_pred"][:n_crop], color="#7f7f7f", linewidth=1.0,
                label=f"Baseline{baseline_tag}", alpha=0.7, zorder=1)

        calibrations = s.get("calibrations", [])
        for i, c in enumerate(calibrations):
            color = _CALIB_COLOR_CYCLE[i % len(_CALIB_COLOR_CYCLE)]
            ax.plot(t, c["pred"][:n_crop], color=color, linewidth=1.1,
                    label=f"+{c['label']} (calibrated)", alpha=0.85, zorder=2 + i)

        title = s.get("subject_id", f"Subject {row}")
        bm = s.get("baseline_metrics")
        if bm:
            title += "\n" + _row(f"Baseline{baseline_tag}", bm)
            for c in calibrations:
                if c.get("metrics"):
                    title += "\n" + _row(f"+{c['label']}", c["metrics"])
        true_dx = _format_true_probs(s.get("true_probs_mean"), superclasses)
        if true_dx:
            title += "\n" + true_dx
        ax.set_title(title, fontsize=8.5, family="monospace", loc="left")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Amplitude (z)", fontsize=8)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(suptitle or "Real ECG vs Baseline vs Calibration Objectives", fontsize=10.5, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


_MODEL_MARKER_CYCLE = ["o", "s", "^", "D", "P"]

_METRIC_AXIS_LABELS = {
    "prd": "PRD (%)",
    "rmse": "RMSE",
    "emd": "EMD",
    "ks_stat": "KS statistic",
    "pearson_r": "Pearson $r$",
    "beat_timing_mae": "Beat-timing MAE (s)",
    "diag_kl": "Diagnostic KL",
    "flip_rate": "Top-1 flip rate",
}


def plot_distortion_perception_plane(
    models: List[Dict],
    x_metric: str = "prd",
    y_metric: str = "emd",
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    annotate_baseline: bool = False,
) -> None:
    """Distortion-vs-perception scatter (Blau & Michaeli style), with arrows
    from each model's baseline to its calibrated variants.

    Locked design per project_distortion_perception_plot_design: x_metric/
    y_metric default to PRD/EMD only -- diag_kl/flip_rate are deliberately
    NOT plotted here (they stay in the per-class diagnostic tables, where
    their CD-specific finding isn't lost to aggregation).

    Args:
        models: list of dicts, one per model variant, each with:
            "label":   display name (e.g. "ReHeartNet + CLEF").
            "baseline": dict with x_metric/y_metric keys (mean values) and
                optionally f"{metric}_ci" keys (95% CI margin, drawn as
                error bars).
            "calibrations": list of dicts, one per calibration objective,
                each with "label" (e.g. "MSE") plus the same x_metric/
                y_metric (+ optional _ci) keys as baseline.
        x_metric, y_metric: keys to read from each point's dict.
        save_path: defaults to results/figures/distortion_perception_plane.png.
        title:     figure title. Defaults to a generic description.
        annotate_baseline: if True, label each baseline point with its
            model name directly on the plot. Off by default -- the "Model"
            legend already encodes this via marker shape, and text labels
            tend to collide with nearby points once several models are
            plotted together.
    """
    if save_path is None:
        save_path = os.path.join(_FIG_DIR, "distortion_perception_plane.png")
    _ensure_fig_dir(save_path)

    x_label = _METRIC_AXIS_LABELS.get(x_metric, x_metric)
    y_label = _METRIC_AXIS_LABELS.get(y_metric, y_metric)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.set_facecolor("white")

    model_handles: List = []
    calib_handles: Dict[str, object] = {}

    for mi, m in enumerate(models):
        marker = _MODEL_MARKER_CYCLE[mi % len(_MODEL_MARKER_CYCLE)]
        bx, by = m["baseline"][x_metric], m["baseline"][y_metric]
        bx_err = m["baseline"].get(f"{x_metric}_ci")
        by_err = m["baseline"].get(f"{y_metric}_ci")
        ax.errorbar(bx, by, xerr=bx_err, yerr=by_err, fmt=marker, color="#8c8c8c",
                    markersize=10, markeredgecolor="black", markeredgewidth=0.9,
                    ecolor="#bbbbbb", elinewidth=1.0, capsize=2.5, capthick=1.0,
                    alpha=0.95, zorder=4)
        if annotate_baseline:
            ax.annotate(m["label"], (bx, by), textcoords="offset points",
                        xytext=(7, 7), fontsize=7.5, color="#333333",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        model_handles.append(plt.Line2D([], [], marker=marker, linestyle="none",
                                         color="#8c8c8c", markeredgecolor="black",
                                         markersize=9, label=m["label"]))

        for ci, c in enumerate(m.get("calibrations", [])):
            color = _CALIB_COLOR_CYCLE[ci % len(_CALIB_COLOR_CYCLE)]
            cx, cy = c[x_metric], c[y_metric]
            cx_err = c.get(f"{x_metric}_ci")
            cy_err = c.get(f"{y_metric}_ci")
            ax.annotate("", xy=(cx, cy), xytext=(bx, by),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                        alpha=0.55, shrinkA=8, shrinkB=8,
                                        connectionstyle="arc3,rad=0.08"), zorder=2)
            ax.errorbar(cx, cy, xerr=cx_err, yerr=cy_err, fmt=marker, color=color,
                        markersize=10, markeredgecolor="black", markeredgewidth=0.9,
                        ecolor=color, elinewidth=1.0, capsize=2.5, capthick=1.0,
                        alpha=0.5, zorder=3)
            ax.scatter([cx], [cy], marker=marker, s=80, facecolor=color,
                       edgecolor="black", linewidth=0.9, zorder=5)
            if c["label"] not in calib_handles:
                calib_handles[c["label"]] = plt.Line2D(
                    [], [], marker="s", linestyle="none", color=color,
                    markersize=8, label=f"+{c['label']}")

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title or "Distortion-Perception Plane: Baseline vs Calibration Objectives",
                 fontsize=12, pad=10)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9.5)

    leg1 = ax.legend(handles=model_handles, title="Model", loc="upper left",
                      bbox_to_anchor=(1.01, 1.0), fontsize=8.5, title_fontsize=9,
                      frameon=False)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=list(calib_handles.values()), title="Calibration",
                      loc="upper left", bbox_to_anchor=(1.01, 1.0 - 0.09 * (len(model_handles) + 1.6)),
                      fontsize=8.5, title_fontsize=9, frameon=False)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", bbox_extra_artists=[leg1, leg2])
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
        metric_names = ["rmse", "prd", "pearson_r", "bce", "emd", "ks_stat", "ks_pvalue", "beat_timing_mae"]

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
