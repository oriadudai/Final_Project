"""Diagnose whether batch_size=1 (Lee et al. spec) is the cause of the
"stuck near constant output" optimization failure observed for
reheartnet_original/huber (see memory: project_train_stuck_near_constant_output).

Hypothesis: with Adam + batch_size=1, each step's gradient reflects a single
window's loss and is dominated by sample-to-sample noise. The EMA of
gradients (m_hat, Adam's update numerator) trends toward 0 as positive/
negative per-sample contributions cancel, while the EMA of squared gradients
(v_hat, the denominator) stays large -> effective step size m_hat/sqrt(v_hat)
collapses toward 0 regardless of lr=1e-2, leaving the network parked near its
post-epoch-1 state.

This script trains ONE fold (default: fold 6, the longest "clean"/zero-nan
production run, 297 epochs flat at MSE~0.92-0.94) for a short number of
epochs, with everything else identical to reheartnet_original (lr=1e-2,
H=32, mse loss, 4s windows, no overlap, bandpass=True, train-only phase
alignment) but with batch_size as a free parameter. Compare:

    python scripts/diag_batch_size_effect.py --fold 6 --batch-size 1  --epochs 100  # control
    python scripts/diag_batch_size_effect.py --fold 6 --batch-size 8  --epochs 100  # test
    python scripts/diag_batch_size_effect.py --fold 6 --batch-size 16 --epochs 100  # test

If batch_size=8/16 shows train loss starting a sustained decrease while
batch_size=1 stays flat (matching the production fold06 trajectory), that
confirms the Adam/batch_size=1 gradient-noise mechanism.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data_loader import build_group_fold, split_train_val
from core.train import train_fold


def quick_metrics(model, loader, device):
    model.eval()
    t_all, p_all = [], []
    with torch.no_grad():
        for ppg, ecg in loader:
            pred = model(ppg.to(device)).squeeze(-1).cpu().numpy()
            t_all.append(ecg.squeeze(-1).numpy())
            p_all.append(pred)
    t = np.concatenate(t_all).ravel()
    p = np.concatenate(p_all).ravel()
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    denom = float((t ** 2).sum())
    prd = float(np.sqrt(((t - p) ** 2).sum() / denom) * 100) if denom > 1e-12 else float("nan")
    r = float(np.corrcoef(t, p)[0, 1]) if t.std() > 1e-8 and p.std() > 1e-8 else float("nan")
    return {"rmse": rmse, "prd": prd, "pearson_r": r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--hidden-size", type=int, default=32)
    ap.add_argument("--loss-type", type=str, default="mse")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold-assignments", type=str, default="results/fold_assignments.json")
    ap.add_argument("--output-dir", type=str, default="results/diag_batch_size")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.fold_assignments) as f:
        assignments = json.load(f)
    fold_key = f"fold_{args.fold:02d}"
    train_subs = assignments[fold_key]["train"]
    test_subs = assignments[fold_key]["test"]
    print(f"Fold {args.fold}: {len(train_subs)} train subjects, {len(test_subs)} test subjects")
    print(f"  test subjects: {test_subs}")

    # Same preprocessing as reheartnet_original: 4s windows, no overlap,
    # bandpass=True, train-only phase alignment.
    train_ds, _test_ds = build_group_fold(
        train_subs, test_subs, apply_align=True,
        window_sec=4.0, overlap_frac=0.0, apply_bandpass=True,
    )
    train_inner, val_ds = split_train_val(train_ds, val_fraction=0.1)
    print(f"  train_inner={len(train_inner)} windows  val={len(val_ds)} windows")

    torch.manual_seed(args.seed)

    train_loader = DataLoader(train_inner, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    run_name = f"bs{args.batch_size}_fold{args.fold:02d}"
    ckpt_dir = os.path.join(args.output_dir, run_name, "checkpoints")

    print(f"\n  loss={args.loss_type}  lr={args.lr}  H={args.hidden_size}  "
          f"batch_size={args.batch_size}  epochs={args.epochs}\n")

    model, history = train_fold(
        fold_idx=args.fold,
        fold_subjects=test_subs,
        train_loader=train_loader,
        val_loader=val_loader,
        clef_encoder=None,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        checkpoint_dir=ckpt_dir,
        use_wandb=False,
        model_name="reheartnet",
        loss_type=args.loss_type,
        lr_schedule="linear_decay",
        early_stop_patience=None,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{run_name}_history.json")
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)

    val_metrics = quick_metrics(model, val_loader, device)

    print(f"\n=== Summary ({run_name}) ===")
    print(f"train: epoch1={history['train'][0]:.5f}  epoch{len(history['train'])}={history['train'][-1]:.5f}  "
          f"min={min(history['train']):.5f}")
    print(f"val:   epoch1={history['val'][0]:.5f}  epoch{len(history['val'])}={history['val'][-1]:.5f}  "
          f"min={min(history['val']):.5f}")
    print(f"final (best-checkpoint) val metrics: "
          f"RMSE={val_metrics['rmse']:.4f}  PRD={val_metrics['prd']:.2f}%  r={val_metrics['pearson_r']:.4f}")
    print(f"Saved loss history to {out_path}")


if __name__ == "__main__":
    main()
