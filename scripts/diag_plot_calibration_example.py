"""Plot real ECG vs baseline (population) vs calibrated (per-subject)
reconstructions for ONE subject, illustrating the full-N=53 win-win result
in [[project_clef_calibration_emd_tradeoff]] (reheartnet_original, lr=Optuna,
--calib-loss composite -- see diag_calib_loss_ablation.py).

Reproduces that script's per-subject protocol for a single (model, fold,
subject) triple, but additionally keeps the raw eval-slice ECG arrays so they
can be plotted with core.visualization.plots.plot_calibration_comparison.

Defaults to bidmc09 (fold 1), the subject with the best PRD/r improvement
among the full-N=53 win-win subjects (PRD 100.28%->47.81%, r +0.141->+0.879).
bidmc15 (fold 7) is a strong alternative with the best absolute EMD/beat-MAE
after calibration (--subject bidmc15 --fold 7).

Usage:
    python scripts/diag_plot_calibration_example.py
    python scripts/diag_plot_calibration_example.py --subject bidmc15 --fold 7
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.config as config
from core.data_loader import build_test_dataset
from core.models.baselines import get_model
from core.train import train_one_epoch
from core.losses.composite_loss import ClinicalCompositeLoss, load_clef_encoder
from core.visualization.plots import plot_calibration_comparison


def predict_all(model, loader, device):
    model.eval()
    t_all, p_all = [], []
    with torch.no_grad():
        for ppg, ecg in loader:
            pred = model(ppg.to(device)).squeeze(-1).cpu().numpy()
            t_all.append(ecg.squeeze(-1).numpy())
            p_all.append(pred)
    return np.concatenate(t_all, axis=0), np.concatenate(p_all, axis=0)


def quick_metrics_from_arrays(t_stack, p_stack):
    t, p = t_stack.ravel(), p_stack.ravel()
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    denom = float((t ** 2).sum())
    prd = float(np.sqrt(((t - p) ** 2).sum() / denom) * 100) if denom > 1e-12 else float("nan")
    r = float(np.corrcoef(t, p)[0, 1]) if t.std() > 1e-8 and p.std() > 1e-8 else float("nan")
    return {"rmse": rmse, "prd": prd, "pearson_r": r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default=os.path.join("results", "comparison_reheartnet_optunalr"))
    ap.add_argument("--model", type=str, default="reheartnet_original")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--subject", type=str, default="bidmc09")
    ap.add_argument("--fold-assignments", type=str, default=None,
                    help="Defaults to <results-dir>/fold_assignments.json")
    ap.add_argument("--window-sec", type=float, default=None)
    ap.add_argument("--overlap-frac", type=float, default=0.5)
    ap.add_argument("--apply-bandpass", action="store_true", default=False)
    ap.add_argument("--calib-frac", type=float, default=0.2)
    ap.add_argument("--calib-epochs", type=int, default=20)
    ap.add_argument("--calib-lr", type=float, default=1e-4)
    ap.add_argument("--calib-loss", type=str, default="composite",
                    choices=["composite", "huber", "mse", "clef-only"])
    ap.add_argument("--huber-delta", type=float, default=1.71377251715061)
    ap.add_argument("--lambda-clinical", type=float, default=None)
    ap.add_argument("--clef-path", type=str, default=None)
    ap.add_argument("--clef-dir", type=str, default=config.CLEF_CHECKPOINT_DIR)
    ap.add_argument("--clef-size", type=str, default="auto", choices=["auto", "small", "medium", "large"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-samples", type=int, default=3,
                    help="Number of eval-slice windows to plot (chosen evenly spaced).")
    ap.add_argument("--output-dir", type=str, default=os.path.join("results", "figures"))
    args = ap.parse_args()

    device = config.DEVICE
    print(f"Device: {device}")

    fold_assignments_path = args.fold_assignments or os.path.join(args.results_dir, "fold_assignments.json")
    with open(fold_assignments_path) as f:
        assignments = json.load(f)
    test_subs = assignments[f"fold_{args.fold:02d}"]["test"]
    if args.subject not in test_subs:
        raise ValueError(f"{args.subject} not in fold {args.fold} test set: {test_subs}")

    ckpt_path = os.path.join(args.results_dir, args.model, "checkpoints", f"reheartnet_fold_{args.fold:02d}_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    hidden_size = ckpt.get("hidden_size", config.HIDDEN_SIZE)
    print(f"Loaded checkpoint: {ckpt_path}  (hidden_size={hidden_size})")

    if args.calib_loss in ("composite", "clef-only"):
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
        huber_weight = 0.0 if args.calib_loss == "clef-only" else 1.0
        criterion = ClinicalCompositeLoss(
            clef_encoder, args.lambda_clinical, args.huber_delta, huber_weight=huber_weight).to(device)
        print(f"Calibration loss: {huber_weight}*Huber(delta={args.huber_delta}) + "
              f"{args.lambda_clinical}*CLEF feature loss")
    elif args.calib_loss == "mse":
        criterion = nn.MSELoss()
        print("Calibration loss: MSE")
    else:
        criterion = nn.HuberLoss(delta=args.huber_delta)
        print(f"Calibration loss: Huber(delta={args.huber_delta})")

    ds = build_test_dataset([args.subject], window_sec=args.window_sec,
                             overlap_frac=args.overlap_frac,
                             apply_bandpass=args.apply_bandpass)
    n = len(ds)
    n_calib = max(1, min(n - 1, int(round(n * args.calib_frac))))
    n_eval = n - n_calib
    print(f"{args.subject}: n_windows={n}  n_calib={n_calib}  n_eval={n_eval}")

    calib_ds = Subset(ds, range(0, n_calib))
    eval_ds = Subset(ds, range(n_calib, n))
    calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    torch.manual_seed(args.seed)
    model = get_model("reheartnet", hidden_size=hidden_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    true_eval, baseline_pred = predict_all(model, eval_loader, device)
    baseline = quick_metrics_from_arrays(true_eval, baseline_pred)
    print(f"  baseline  : RMSE={baseline['rmse']:.4f}  PRD={baseline['prd']:6.2f}%  r={baseline['pearson_r']:+.4f}")

    optimizer = optim.Adam(model.parameters(), lr=args.calib_lr)
    for _ in range(args.calib_epochs):
        train_one_epoch(model, calib_loader, criterion, optimizer, device)

    _, calibrated_pred = predict_all(model, eval_loader, device)
    calibrated = quick_metrics_from_arrays(true_eval, calibrated_pred)
    print(f"  calibrated: RMSE={calibrated['rmse']:.4f}  PRD={calibrated['prd']:6.2f}%  r={calibrated['pearson_r']:+.4f}")

    save_path = os.path.join(args.output_dir, f"calibration_comparison_{args.subject}_fold{args.fold:02d}.png")
    plot_calibration_comparison(true_eval, baseline_pred, calibrated_pred, fs=config.FS,
                                 n_samples=args.n_samples, save_path=save_path, subject_id=args.subject)
    print(f"\nSaved plot to {save_path}")


if __name__ == "__main__":
    main()
