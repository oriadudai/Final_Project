"""Plot real ECG vs baseline vs ALL calibration objectives (MSE, Huber,
Composite), for ONE subject, across ALL THREE model variants -- the
transpose of diag_plot_calibration_multi_loss.py (which fixes one model
and varies subjects; this fixes one subject and varies models).

One panel per model (reheartnet_original, reheartnet_huber, arch_reheartnet),
each showing the same subject/window: real ECG, the uncalibrated baseline,
and all three calibration objectives. Useful for picking a subject with a
notable pathology classification (e.g. high Conduction-Defect probability
on the real ECG) and seeing how every model x calibration combination
reconstructs it.

Reuses run_subject() from diag_plot_calibration_multi_loss.py unchanged --
that function already reads the model/checkpoint to use from args.model/
args.results_dir, so this script just calls it once per model with the
SAME subject/fold and collects the results into one multi-panel figure.

Usage:
    python scripts/diag_plot_calibration_multi_model.py
    python scripts/diag_plot_calibration_multi_model.py --subject bidmc15 --fold 7
"""
import argparse
import copy
import json
import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.config as config
from core.losses.composite_loss import ClinicalCompositeLoss, load_clef_encoder
from core.visualization.plots import plot_calibration_multi_loss
from diag_calib_loss_ablation import _load_diag_clf
from diag_plot_calibration_multi_loss import run_subject

_MODEL_DISPLAY = {
    "reheartnet_original": "ReHeartNet (original)",
    "reheartnet_huber":    "ReHeartNet + Huber",
    "arch_reheartnet":     "ReHeartNet + CLEF",
}
_BASELINE_LOSS_LABEL = {
    "reheartnet_original": "MSE",
    "reheartnet_huber":    "Huber",
    "arch_reheartnet":     "Huber + CLEF",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default=os.path.join("results", "comparison_reheartnet_optunalr"))
    ap.add_argument("--models", type=str, default="reheartnet_original,reheartnet_huber,arch_reheartnet")
    ap.add_argument("--subject", type=str, default="bidmc15",
                    help="Subject to plot across all models (default: bidmc15, a clean "
                         "CD-dominant subject, true_probs CD~0.97 on the real ECG).")
    ap.add_argument("--fold", type=int, default=7,
                    help="Fold containing --subject in its test split (shared across all "
                         "three model variants via the common fold_assignments.json).")
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
    ap.add_argument("--diagnostic-classifier", type=str,
                    default=os.path.join("checkpoints", "ptbxl_diagnostic_classifier.pt"))
    ap.add_argument("--output", type=str, default=None,
                    help="Defaults to results/figures/calibration_multi_model_<subject>.png")
    args = ap.parse_args()

    if args.output is None:
        args.output = os.path.join("results", "figures", f"calibration_multi_model_{args.subject}.png")

    device = config.DEVICE
    print(f"Device: {device}")

    models = [m.strip() for m in args.models.split(",")]

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
    print(f"Calibration objectives: {requested_labels}")

    diag_clf, diag_superclasses = None, None
    if os.path.exists(args.diagnostic_classifier):
        diag_clf, diag_superclasses = _load_diag_clf(args.diagnostic_classifier, device, clef_encoder=clef_encoder)
        print(f"Diagnostic classifier loaded: {args.diagnostic_classifier}  (superclasses={diag_superclasses})")
    else:
        print(f"WARNING: diagnostic classifier not found at {args.diagnostic_classifier}; "
              f"KL/flip will be omitted from the plot.")

    results = []
    for model_key in models:
        # Criteria must be rebuilt per model since some hold stateful nn.Module instances.
        criteria = {lbl: available[lbl]() for lbl in requested_labels}
        model_args = copy.copy(args)
        model_args.model = model_key
        print(f"\n{'='*60}\n  Model: {_MODEL_DISPLAY.get(model_key, model_key)}\n{'='*60}")
        entry = run_subject(model_args, args.subject, args.fold, criteria, diag_clf, device)
        # plot_calibration_multi_loss applies one global baseline_loss_label to every
        # panel; since each panel here is a DIFFERENT model (trained with a different
        # loss), bake the per-model label into subject_id instead of relying on that.
        display = _MODEL_DISPLAY.get(model_key, model_key)
        trained_as = _BASELINE_LOSS_LABEL.get(model_key, model_key)
        entry["subject_id"] = f"{display}  [trained: {trained_as}]"
        results.append(entry)

    suptitle = (f"{args.subject} (fold {args.fold}): Real ECG vs Baseline vs "
                f"{', '.join(requested_labels)} Calibration, across all three model variants")
    plot_calibration_multi_loss(results, fs=config.FS, save_path=args.output, crop_sec=args.crop_sec,
                                 baseline_loss_label=None, suptitle=suptitle,
                                 superclasses=diag_superclasses)
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()
