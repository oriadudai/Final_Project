"""Plot real ECG vs baseline vs two calibration objectives, illustrating the
perception-distortion calibration asymmetry of [[project_clef_calibration_emd_tradeoff]]
(see also diag_calib_loss_ablation.py's full 3x3 ablation and
core/visualization/plots.py:plot_calibration_asymmetry).

For a CLEF-trained model (arch_reheartnet), distortion-only calibration (MSE)
produces a large pointwise improvement but actively regresses EMD/KS below
baseline (negative-sum regime); composite calibration (Huber + CLEF term)
gives a smaller pointwise gain but preserves EMD/KS (coopetitive regime).
This script calibrates the SAME baseline checkpoint twice per subject -- once
with each loss -- and plots all four traces (real, baseline, +MSE, +Composite)
on one panel so the asymmetry is visible directly in the waveform, not just
in the aggregate metrics table.

Default subjects were chosen by scanning all 35 arch_reheartnet test subjects
for the largest (MSE EMD regression) + (Composite EMD preservation) score:
  - bidmc06 (fold 0): EMD 0.038->0.325 under MSE (8.6x worse) vs ->0.038 under
                      Composite (preserved); KS 0.152->0.863 vs ->0.169.
  - bidmc09 (fold 1): EMD 0.188->0.452 under MSE (2.4x worse) vs ->0.167 under
                      Composite (slightly better than baseline); KS
                      0.191->0.734 vs ->0.181. Same subject already used in
                      calibration_comparison_multi.png for reheartnet_original.

Usage:
    python scripts/diag_plot_calibration_asymmetry.py
    python scripts/diag_plot_calibration_asymmetry.py --subjects bidmc18,bidmc32 --folds 0,4
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.config as config
from core.data_loader import build_test_dataset
from core.metrics.clinical_metrics import compute_beat_timing_mae, compute_emd, compute_ks
from core.models.baselines import get_model
from core.train import train_one_epoch
from core.losses.composite_loss import ClinicalCompositeLoss, load_clef_encoder
from core.visualization.plots import plot_calibration_asymmetry
from diag_subject_calibration import chronological_calib_eval_split
from diag_calib_loss_ablation import _load_diag_clf, _run_diag

# Models trained via run_cv.py: checkpoints live at the fixed config.CHECKPOINT_DIR,
# named only by model_name + fold (mirrors diag_calib_loss_ablation.py's EXTRA_MODELS).
EXTRA_MODELS = {
    "arch_reheartnet": {
        "ckpt_path": lambda fold_idx: os.path.join(
            config.CHECKPOINT_DIR, f"reheartnet_fold_{fold_idx:02d}_best.pt"),
    },
}


def predict_all(model, loader, device):
    model.eval()
    t_all, p_all = [], []
    with torch.no_grad():
        for ppg, ecg in loader:
            pred = model(ppg.to(device)).squeeze(-1).cpu().numpy()
            t_all.append(ecg.squeeze(-1).numpy())
            p_all.append(pred)
    return np.concatenate(t_all, axis=0), np.concatenate(p_all, axis=0)


def quick_metrics_from_arrays(t_stack, p_stack, fs):
    t, p = t_stack.ravel(), p_stack.ravel()
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    denom = float((t ** 2).sum())
    prd = float(np.sqrt(((t - p) ** 2).sum() / denom) * 100) if denom > 1e-12 else float("nan")
    r = float(np.corrcoef(t, p)[0, 1]) if t.std() > 1e-8 and p.std() > 1e-8 else float("nan")
    emd = compute_emd(t_stack, p_stack, fs=fs)
    ks_stat, _ = compute_ks(t_stack, p_stack, fs=fs)
    beat_mae = compute_beat_timing_mae(t_stack, p_stack, fs=fs)
    return {"rmse": rmse, "prd": prd, "pearson_r": r,
            "emd": emd, "ks_stat": ks_stat, "beat_timing_mae": beat_mae}


def _ckpt_path(args, fold):
    if args.model in EXTRA_MODELS:
        return EXTRA_MODELS[args.model]["ckpt_path"](fold)
    return os.path.join(args.results_dir, args.model, "checkpoints", f"reheartnet_fold_{fold:02d}_best.pt")


def calibrate_copy(base_state_dict, hidden_size, calib_loader, criterion, args, device):
    """Fresh copy of the base checkpoint, fine-tuned on calib_loader with criterion."""
    model = get_model("reheartnet", hidden_size=hidden_size).to(device)
    model.load_state_dict(copy.deepcopy(base_state_dict))
    optimizer = optim.Adam(model.parameters(), lr=args.calib_lr)
    for _ in range(args.calib_epochs):
        train_one_epoch(model, calib_loader, criterion, optimizer, device)
    return model


def _add_diag_metrics(metrics, true_arr, pred_arr, diag_clf, device):
    """In-place: add diag_kl / flip_rate to a quick_metrics_from_arrays() dict."""
    if diag_clf is None:
        return metrics
    kl, flip_rate, _, _ = _run_diag(true_arr, pred_arr, diag_clf, device)
    metrics["diag_kl"] = kl
    metrics["flip_rate"] = flip_rate
    return metrics


def run_subject(args, subject, fold, mse_criterion, composite_criterion, diag_clf, device):
    fold_assignments_path = args.fold_assignments or os.path.join(args.results_dir, "fold_assignments.json")
    with open(fold_assignments_path) as f:
        assignments = json.load(f)
    test_subs = assignments[f"fold_{fold:02d}"]["test"]
    if subject not in test_subs:
        raise ValueError(f"{subject} not in fold {fold} test set: {test_subs}")

    ckpt_path = _ckpt_path(args, fold)
    ckpt = torch.load(ckpt_path, map_location=device)
    hidden_size = ckpt.get("hidden_size", config.HIDDEN_SIZE)
    print(f"Loaded checkpoint: {ckpt_path}  (hidden_size={hidden_size})")

    ds = build_test_dataset([subject], window_sec=args.window_sec,
                             overlap_frac=args.overlap_frac,
                             apply_bandpass=args.apply_bandpass)
    n = len(ds)
    calib_ds, eval_ds, n_calib, n_eval = chronological_calib_eval_split(
        ds, args.calib_frac, args.overlap_frac)
    print(f"{subject} (fold{fold:02d}): n_windows={n}  n_calib={n_calib}  n_eval={n_eval}")

    calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    base_model = get_model("reheartnet", hidden_size=hidden_size).to(device)
    base_model.load_state_dict(ckpt["model_state_dict"])
    true_eval, baseline_pred = predict_all(base_model, eval_loader, device)
    baseline = quick_metrics_from_arrays(true_eval, baseline_pred, config.FS)
    _add_diag_metrics(baseline, true_eval, baseline_pred, diag_clf, device)
    print(f"  baseline  : RMSE={baseline['rmse']:.4f}  PRD={baseline['prd']:6.2f}%  r={baseline['pearson_r']:+.3f}  "
          f"EMD={baseline['emd']:.4f}  KS={baseline['ks_stat']:.3f}"
          + (f"  KL={baseline['diag_kl']:.3f}  flip={baseline['flip_rate']:.3f}" if diag_clf is not None else ""))

    torch.manual_seed(args.seed)
    model_a = calibrate_copy(ckpt["model_state_dict"], hidden_size, calib_loader, mse_criterion, args, device)
    _, calib_a_pred = predict_all(model_a, eval_loader, device)
    calib_a = quick_metrics_from_arrays(true_eval, calib_a_pred, config.FS)
    _add_diag_metrics(calib_a, true_eval, calib_a_pred, diag_clf, device)
    print(f"  +{args.calib_a_label:<9}: RMSE={calib_a['rmse']:.4f}  PRD={calib_a['prd']:6.2f}%  r={calib_a['pearson_r']:+.3f}  "
          f"EMD={calib_a['emd']:.4f}  KS={calib_a['ks_stat']:.3f}"
          + (f"  KL={calib_a['diag_kl']:.3f}  flip={calib_a['flip_rate']:.3f}" if diag_clf is not None else ""))

    torch.manual_seed(args.seed)
    model_b = calibrate_copy(ckpt["model_state_dict"], hidden_size, calib_loader, composite_criterion, args, device)
    _, calib_b_pred = predict_all(model_b, eval_loader, device)
    calib_b = quick_metrics_from_arrays(true_eval, calib_b_pred, config.FS)
    _add_diag_metrics(calib_b, true_eval, calib_b_pred, diag_clf, device)
    print(f"  +{args.calib_b_label:<9}: RMSE={calib_b['rmse']:.4f}  PRD={calib_b['prd']:6.2f}%  r={calib_b['pearson_r']:+.3f}  "
          f"EMD={calib_b['emd']:.4f}  KS={calib_b['ks_stat']:.3f}"
          + (f"  KL={calib_b['diag_kl']:.3f}  flip={calib_b['flip_rate']:.3f}" if diag_clf is not None else ""))

    window_idx = n_eval // 2
    return {
        "subject_id": subject,
        "true_ecg": true_eval[window_idx],
        "baseline_pred": baseline_pred[window_idx],
        "calib_a_pred": calib_a_pred[window_idx],
        "calib_a_label": args.calib_a_label,
        "calib_b_pred": calib_b_pred[window_idx],
        "calib_b_label": args.calib_b_label,
        "baseline_metrics": baseline,
        "calib_a_metrics": calib_a,
        "calib_b_metrics": calib_b,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default=os.path.join("results", "comparison_reheartnet_optunalr"))
    ap.add_argument("--model", type=str, default="arch_reheartnet")
    ap.add_argument("--subjects", type=str, default="bidmc06,bidmc09")
    ap.add_argument("--folds", type=str, default="0,1")
    ap.add_argument("--fold-assignments", type=str, default=None,
                    help="Defaults to <results-dir>/fold_assignments.json")
    ap.add_argument("--window-sec", type=float, default=None)
    ap.add_argument("--overlap-frac", type=float, default=0.5)
    ap.add_argument("--apply-bandpass", action="store_true", default=False)
    ap.add_argument("--calib-frac", type=float, default=0.2)
    ap.add_argument("--calib-epochs", type=int, default=20)
    ap.add_argument("--calib-lr", type=float, default=1e-4)
    ap.add_argument("--huber-delta", type=float, default=1.71377251715061)
    ap.add_argument("--lambda-clinical", type=float, default=None)
    ap.add_argument("--clef-path", type=str, default=None)
    ap.add_argument("--clef-dir", type=str, default=config.CLEF_CHECKPOINT_DIR)
    ap.add_argument("--clef-size", type=str, default="auto", choices=["auto", "small", "medium", "large"])
    ap.add_argument("--calib-a-label", type=str, default="MSE")
    ap.add_argument("--calib-b-label", type=str, default="Composite")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crop-sec", type=float, default=4.0,
                    help="Crop window to this many seconds for readability (default 4s; "
                         "pass --crop-sec 0 for the full window, e.g. 10s for CLEF's window config).")
    ap.add_argument("--baseline-loss-label", type=str, default=None,
                    help="Label for the baseline model's training loss, shown in the plot "
                         "(default: auto-detected from --model: 'Huber + CLEF' for arch_reheartnet, "
                         "'MSE' for reheartnet_original, 'Huber' for reheartnet_huber).")
    ap.add_argument("--diagnostic-classifier", type=str,
                    default=os.path.join("checkpoints", "ptbxl_diagnostic_classifier.pt"),
                    help="Path to the PTB-XL diagnostic classifier checkpoint, used to compute "
                         "diag_kl and flip_rate. If not found, KL/flip are omitted from the plot.")
    ap.add_argument("--output", type=str, default=os.path.join("results", "figures", "calibration_asymmetry.png"))
    args = ap.parse_args()

    device = config.DEVICE
    print(f"Device: {device}")

    subjects = args.subjects.split(",")
    folds = [int(f) for f in args.folds.split(",")]
    if len(subjects) != len(folds):
        raise ValueError("--subjects and --folds must have the same length")

    if args.baseline_loss_label is None:
        args.baseline_loss_label = {
            "arch_reheartnet": "Huber + CLEF",
            "reheartnet_original": "MSE",
            "reheartnet_huber": "Huber",
        }.get(args.model, args.model)
        print(f"baseline_loss_label auto-detected from --model={args.model}: {args.baseline_loss_label}")

    mse_criterion = nn.MSELoss()

    if args.lambda_clinical is None:
        best_hp_path = os.path.join("results", "best_hyperparams.json")
        if os.path.exists(best_hp_path):
            with open(best_hp_path) as f:
                args.lambda_clinical = float(json.load(f)["lambda_clinical"])
            print(f"lambda_clinical read from {best_hp_path}: {args.lambda_clinical}")
        else:
            args.lambda_clinical = config.LAMBDA_CLINICAL
            print(f"{best_hp_path} not found; lambda_clinical defaults to "
                  f"config.LAMBDA_CLINICAL={args.lambda_clinical}")
    if args.clef_size == "auto":
        args.clef_size = "medium" if device.type == "cuda" else "small"
        print(f"CLEF size auto-selected: {args.clef_size}")
    if args.clef_path is None:
        args.clef_path = os.path.join(args.clef_dir, f"clef_{args.clef_size}.ckpt")
        print(f"CLEF path auto-set: {args.clef_path}")
    print(f"Loading CLEF encoder ({args.clef_size}) ...")
    clef_encoder = load_clef_encoder(args.clef_path, args.clef_size, device)
    composite_criterion = ClinicalCompositeLoss(
        clef_encoder, args.lambda_clinical, args.huber_delta, huber_weight=1.0).to(device)
    print(f"Calibration A: {args.calib_a_label} | Calibration B: {args.calib_b_label} = "
          f"Huber(delta={args.huber_delta}) + {args.lambda_clinical}*CLEF feature loss")

    diag_clf = None
    if os.path.exists(args.diagnostic_classifier):
        diag_clf, diag_superclasses = _load_diag_clf(args.diagnostic_classifier, device, clef_encoder=clef_encoder)
        print(f"Diagnostic classifier loaded: {args.diagnostic_classifier}  (superclasses={diag_superclasses})")
    else:
        print(f"WARNING: diagnostic classifier not found at {args.diagnostic_classifier}; "
              f"KL/flip will be omitted from the plot.")

    results = []
    for subject, fold in zip(subjects, folds):
        results.append(run_subject(args, subject, fold, mse_criterion, composite_criterion, diag_clf, device))

    plot_calibration_asymmetry(results, fs=config.FS, save_path=args.output, crop_sec=args.crop_sec,
                                baseline_loss_label=args.baseline_loss_label)
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()
