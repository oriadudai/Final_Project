"""Plot real ECG vs baseline vs ALL calibration objectives (MSE, Huber,
Composite) for one model variant, generalizing diag_plot_calibration_asymmetry.py
from 2 calibration variants to N.

Run once per model (--model reheartnet_original / reheartnet_huber /
arch_reheartnet) to get the full 3-model x 3-calib-loss = 9-combination
picture for main.tex, using the SAME two subjects (bidmc06 fold 0, bidmc09
fold 1) across all three models for direct comparability -- both folds are
available for all three model variants since they share the same common-window
test-subject assignment (fold_assignments.json), and arch_reheartnet covers
folds 0-4 (see [[project_clef_calibration_emd_tradeoff]] for why folds 5-7
are unavailable for that variant).

Usage:
    python scripts/diag_plot_calibration_multi_loss.py --model reheartnet_original
    python scripts/diag_plot_calibration_multi_loss.py --model reheartnet_huber
    python scripts/diag_plot_calibration_multi_loss.py --model arch_reheartnet
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.config as config
from core.data_loader import build_test_dataset
from core.models.baselines import get_model
from core.losses.composite_loss import ClinicalCompositeLoss, load_clef_encoder
from core.visualization.plots import plot_calibration_multi_loss
from diag_subject_calibration import chronological_calib_eval_split
from diag_calib_loss_ablation import _load_diag_clf
from diag_plot_calibration_asymmetry import (
    EXTRA_MODELS, predict_all, quick_metrics_from_arrays, _ckpt_path,
    calibrate_copy, _add_diag_metrics, _print_peak_coverage,
)
from torch.utils.data import DataLoader


def run_subject(args, subject, fold, criteria, diag_clf, device):
    """criteria: dict {label: criterion}, e.g. {"MSE": nn.MSELoss(), ...}."""
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
    _print_peak_coverage("Real ECG", true_eval, config.FS)
    _print_peak_coverage("Baseline", baseline_pred, config.FS)

    window_idx = n_eval // 2
    calibrations = []
    for label, criterion in criteria.items():
        torch.manual_seed(args.seed)
        model_c = calibrate_copy(ckpt["model_state_dict"], hidden_size, calib_loader, criterion, args, device)
        _, calib_pred = predict_all(model_c, eval_loader, device)
        metrics = quick_metrics_from_arrays(true_eval, calib_pred, config.FS)
        _add_diag_metrics(metrics, true_eval, calib_pred, diag_clf, device)
        print(f"  +{label:<9}: RMSE={metrics['rmse']:.4f}  PRD={metrics['prd']:6.2f}%  r={metrics['pearson_r']:+.3f}  "
              f"EMD={metrics['emd']:.4f}  KS={metrics['ks_stat']:.3f}"
              + (f"  KL={metrics['diag_kl']:.3f}  flip={metrics['flip_rate']:.3f}" if diag_clf is not None else ""))
        _print_peak_coverage(f"+{label}", calib_pred, config.FS)
        calibrations.append({"label": label, "pred": calib_pred[window_idx], "metrics": metrics})

    return {
        "subject_id": subject,
        "true_ecg": true_eval[window_idx],
        "baseline_pred": baseline_pred[window_idx],
        "baseline_metrics": baseline,
        "calibrations": calibrations,
        "true_probs_mean": baseline.get("true_probs_mean"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default=os.path.join("results", "comparison_reheartnet_optunalr"))
    ap.add_argument("--model", type=str, default="arch_reheartnet",
                    choices=["reheartnet_original", "reheartnet_huber", "arch_reheartnet"])
    ap.add_argument("--subjects", type=str, default="bidmc06,bidmc09")
    ap.add_argument("--folds", type=str, default="0,1")
    ap.add_argument("--fold-assignments", type=str, default=None)
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
    ap.add_argument("--calib-losses", type=str, default="MSE,Huber,Composite",
                    help="Comma-separated calibration objectives to apply, in order.")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crop-sec", type=float, default=4.0)
    ap.add_argument("--baseline-loss-label", type=str, default=None)
    ap.add_argument("--diagnostic-classifier", type=str,
                    default=os.path.join("checkpoints", "ptbxl_diagnostic_classifier.pt"))
    ap.add_argument("--output", type=str, default=None,
                    help="Defaults to results/figures/calibration_multi_loss_<model>.png")
    args = ap.parse_args()

    if args.output is None:
        args.output = os.path.join("results", "figures", f"calibration_multi_loss_{args.model}.png")

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

    requested_labels = [s.strip() for s in args.calib_losses.split(",")]
    available = {
        "MSE": lambda: nn.MSELoss(),
        "Huber": lambda: nn.HuberLoss(delta=args.huber_delta),
        "Composite": lambda: ClinicalCompositeLoss(
            clef_encoder, args.lambda_clinical, args.huber_delta, huber_weight=1.0).to(device),
    }
    unknown = [lbl for lbl in requested_labels if lbl not in available]
    if unknown:
        raise ValueError(f"Unknown calibration loss label(s) {unknown}; choose from {list(available)}")
    criteria = {lbl: available[lbl]() for lbl in requested_labels}
    print(f"Calibration objectives: {list(criteria)}")

    diag_clf, diag_superclasses = None, None
    if os.path.exists(args.diagnostic_classifier):
        diag_clf, diag_superclasses = _load_diag_clf(args.diagnostic_classifier, device, clef_encoder=clef_encoder)
        print(f"Diagnostic classifier loaded: {args.diagnostic_classifier}  (superclasses={diag_superclasses})")
    else:
        print(f"WARNING: diagnostic classifier not found at {args.diagnostic_classifier}; "
              f"KL/flip will be omitted from the plot.")

    results = []
    for subject, fold in zip(subjects, folds):
        results.append(run_subject(args, subject, fold, criteria, diag_clf, device))

    suptitle = f"Real ECG vs Baseline ({args.model}, trained: {args.baseline_loss_label}) vs {', '.join(criteria)} Calibration"
    plot_calibration_multi_loss(results, fs=config.FS, save_path=args.output, crop_sec=args.crop_sec,
                                 baseline_loss_label=args.baseline_loss_label, suptitle=suptitle,
                                 superclasses=diag_superclasses)
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()
