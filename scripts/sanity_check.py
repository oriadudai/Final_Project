"""Quick sanity check — runs in ~2-5 minutes on CPU.

Uses 3 BIDMC subjects, 2 manual folds, 3 epochs.
Tests the full pipeline end-to-end:
  preprocessing → data loading → training (Huber+CLEF) → evaluation → plots

Usage:
    python scripts/sanity_check.py
    python scripts/sanity_check.py --clef-path models/clef/clef_small.ckpt
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.config as config

CLEF_PATH   = "models/clef/clef_small.ckpt"
SUBJECTS    = ["bidmc01", "bidmc02", "bidmc03"]
EPOCHS      = 3
BATCH_SIZE  = 16


def separator(title=""):
    w = 55
    if title:
        pad = (w - len(title) - 2) // 2
        print(f"\n{'-'*pad} {title} {'-'*pad}")
    else:
        print("-" * w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clef-path", default=CLEF_PATH)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cpu")
    print(f"Device : {device}")
    print(f"Subjects: {SUBJECTS}")
    print(f"Epochs  : {EPOCHS}  |  Batch size: {BATCH_SIZE}")

    # ── 1. Imports ────────────────────────────────────────────────────────────
    separator("1 · Imports")
    from core.data_loader import build_group_fold, split_train_val
    from core.evaluate import evaluate_fold
    from core.losses.composite_loss import ClinicalCompositeLoss, load_clef_encoder
    from core.models.ptbxl_classifier import get_ptbxl_classifier
    from core.train import train_fold
    from core.visualization.plots import (
        plot_loss_curves, plot_reconstruction_samples, save_results_summary,
    )
    print("  All imports OK")

    # ── 2. CLEF encoder ───────────────────────────────────────────────────────
    separator("2 · CLEF encoder")
    if not os.path.exists(args.clef_path):
        print(f"  CLEF checkpoint not found at {args.clef_path}")
        print("  Falling back to MSE loss for this run.")
        clef_encoder = None
    else:
        clef_encoder = load_clef_encoder(args.clef_path, model_size="small", device=device)
        dummy = torch.randn(2, 1, 5000)
        feat  = clef_encoder(dummy)
        print(f"  Loaded  |  input (2,1,5000) → features {tuple(feat.shape)}")

    ptbxl_clf = get_ptbxl_classifier(pretrained_path=None).to(device)
    print(f"  PTB-XL surrogate classifier ready")

    # ── 3. Preprocessing ─────────────────────────────────────────────────────
    separator("3 · Preprocessing (3 subjects)")
    from src.preprocessing import build_subject_windows
    for subj in SUBJECTS:
        ppg, ecg = build_subject_windows(subj, apply_phase_align=False)
        print(f"  {subj}: ppg={ppg.shape}  ecg={ecg.shape}  "
              f"min/max ecg=[{ecg.min():.2f}, {ecg.max():.2f}]")

    # ── 4. 2-fold CV ─────────────────────────────────────────────────────────
    separator("4 · 2-fold mini CV")
    folds = [
        (["bidmc02", "bidmc03"], ["bidmc01"]),
        (["bidmc01", "bidmc03"], ["bidmc02"]),
    ]

    fold_metrics = []
    os.makedirs("results/sanity_check/figures", exist_ok=True)

    for fold_idx, (train_subs, test_subs) in enumerate(folds):
        print(f"\n  ── Fold {fold_idx} | train={train_subs} | test={test_subs}")

        train_ds, test_ds = build_group_fold(train_subs, test_subs, apply_align=True)
        train_inner, val_ds = split_train_val(train_ds, val_fraction=0.2)

        train_loader = DataLoader(train_inner, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,      batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,     batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        print(f"     windows → train:{len(train_inner)}  val:{len(val_ds)}  test:{len(test_ds)}")

        loss_type = "clef" if clef_encoder is not None else "mse"
        model, history = train_fold(
            fold_idx        = fold_idx,
            fold_subjects   = test_subs,
            train_loader    = train_loader,
            val_loader      = val_loader,
            clef_encoder    = clef_encoder,
            device          = device,
            epochs          = EPOCHS,
            lr              = 1e-3,
            lambda_clinical = 0.1,
            huber_delta     = 1.0,
            hidden_size     = 32,          # smaller than default for speed
            checkpoint_dir  = "results/sanity_check/checkpoints",
            use_wandb       = False,
            early_stop_patience = 999,     # no early stopping in 3 epochs
            loss_type       = loss_type,
        )

        print(f"     train loss history: {[f'{v:.4f}' for v in history['train']]}")
        print(f"     val   loss history: {[f'{v:.4f}' for v in history['val']]}")

        # Loss curve
        plot_loss_curves(
            history["train"], history["val"], fold_idx,
            save_path=f"results/sanity_check/figures/loss_fold{fold_idx}.png",
        )

        # Evaluate
        print(f"     Evaluating ...")
        metrics = evaluate_fold(model, test_loader, ptbxl_clf, device, clef_encoder=clef_encoder)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

        print(f"     PRD     = {metrics['prd']:.3f} %")
        print(f"     Pearson = {metrics['pearson_r']:.3f}")
        print(f"     BCE     = {metrics['bce']:.4f}")
        print(f"     EMD     = {metrics['emd']}")
        print(f"     KS stat = {metrics['ks_stat']}")
        print(f"     Beat MAE= {metrics['beat_timing_mae']}")

        # Reconstruction plot
        model.eval()
        with torch.no_grad():
            ppg_b, ecg_b = next(iter(test_loader))
            pred_b = model(ppg_b.to(device)).cpu()
        plot_reconstruction_samples(
            ecg_b.squeeze(-1).numpy(),
            pred_b.squeeze(-1).numpy(),
            fold_idx=fold_idx,
            n_samples=2,
            save_path=f"results/sanity_check/figures/recon_fold{fold_idx}.png",
        )

    # ── 5. Summary ────────────────────────────────────────────────────────────
    separator("5 · Summary")
    save_results_summary(
        fold_metrics,
        save_path="results/sanity_check/summary.json",
    )

    elapsed = time.time() - t0
    separator()
    print(f"Sanity check PASSED in {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"Figures → results/sanity_check/figures/")
    print(f"Summary → results/sanity_check/summary.json")


if __name__ == "__main__":
    main()
