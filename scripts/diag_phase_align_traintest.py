"""Diagnose whether the train/test phase-alignment mismatch in
build_group_fold (core/data_loader.py) explains the universal test
PRD~100% / pearson_r~0 result observed across ALL 6 loss/architecture
variants (see memory: project_train_test_phase_align_mismatch).

build_group_fold(train_subjects, test_subjects, apply_align=True) applies
phase_align_ppg (circular-roll, see project_phase_align_degrades_corr) to
TRAINING subjects' PPG only -- val_ds is split off this aligned train_ds,
while test_ds is ALWAYS built with apply_align=False. So during a normal
run, val is evaluated on phase-aligned PPG and test on raw/unaligned PPG --
a potential input-distribution shift independent of architecture, loss,
lr, or batch_size.

Hypothesis: if train/val/test all share the SAME (unaligned) PPG
preprocessing (apply_align=False for the training-subjects call too), the
val/test PRD gap should close -- test PRD should drop well below the
"predict near-constant" ~100% baseline, instead of being stuck there
regardless of how well val_loss converges.

This script trains ONE fold and reports RMSE/PRD/pearson_r on three sets:
  - train (the windows actually optimized on)
  - val   (held-out windows from training subjects, same preprocessing as train)
  - test  (held-out subjects, always apply_align=False)

Usage (lr/batch_size/hidden_size default to the Optuna-tuned values already
confirmed to give real, sustained val-loss improvement for
arch_lstm/bilstm/reheartnet -- see project_train_test_phase_align_mismatch):

    # baseline: current production behavior (train/val aligned, test unaligned)
    python scripts/diag_phase_align_traintest.py --fold 6 --apply-align --epochs 100

    # test: train/val/test all unaligned
    python scripts/diag_phase_align_traintest.py --fold 6 --no-apply-align --epochs 100

Compare the printed train/val/test PRD and pearson_r between the two runs.
If --no-apply-align closes the val/test gap (test PRD drops well below
~100%, test r moves toward val r), the train/test preprocessing mismatch is
confirmed as a (likely the) dominant cause of the universal PRD~100%
result.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.config as config
from core.data_loader import build_group_fold, split_train_val
from core.losses.composite_loss import load_clef_encoder
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
    ap.add_argument("--apply-align", action=argparse.BooleanOptionalAction, default=True,
                    help="Apply phase_align_ppg to the training subjects' PPG "
                         "(current production default). Test subjects are NEVER "
                         "aligned, regardless of this flag. Pass --no-apply-align "
                         "to make train/val match test's (unaligned) preprocessing.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.00031452237380888214)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--hidden-size", type=int, default=32)
    ap.add_argument("--loss-type", type=str, default="clef",
                    help="'clef' (default) matches arch_lstm/bilstm/reheartnet's "
                         "actual training objective: Huber + lambda_clinical * "
                         "CLEF-feature L2 ('clinical perceptual loss', NOT a KL "
                         "term -- diag_kl is a separate post-hoc metric, unrelated "
                         "to training). Use 'mse'/'huber' to drop the CLEF term.")
    ap.add_argument("--lambda-clinical", type=float, default=0.9643164565469905)
    ap.add_argument("--huber-delta", type=float, default=1.71377251715061)
    ap.add_argument("--clef-path", type=str, default=None,
                    help="Path to CLEF .ckpt (auto-constructed from --clef-dir/--clef-size if omitted)")
    ap.add_argument("--clef-dir", type=str, default=config.CLEF_CHECKPOINT_DIR)
    ap.add_argument("--clef-size", type=str, default="auto", choices=["auto", "small", "medium", "large"])
    ap.add_argument("--early-stop-patience", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold-assignments", type=str,
                    default="results/comparison_arch/fold_assignments.json")
    ap.add_argument("--output-dir", type=str, default="results/diag_phase_align_traintest")
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

    # Same preprocessing as run_cv.py's architecture-comparison runs (10s
    # windows, 0.5 overlap, no bandpass), with apply_align as the variable
    # under test. test_ds is ALWAYS apply_align=False inside build_group_fold.
    train_ds, test_ds = build_group_fold(train_subs, test_subs, apply_align=args.apply_align)
    train_inner, val_ds = split_train_val(train_ds, val_fraction=0.1)
    print(f"  train_inner={len(train_inner)}  val={len(val_ds)}  test={len(test_ds)} windows")

    torch.manual_seed(args.seed)

    train_loader = DataLoader(train_inner, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    train_eval_loader = DataLoader(train_inner, batch_size=args.batch_size, shuffle=False,
                                    num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    clef_encoder = None
    if args.loss_type == "clef":
        if args.clef_size == "auto":
            args.clef_size = "medium" if device.type == "cuda" else "small"
            print(f"CLEF size auto-selected: {args.clef_size}")
        if args.clef_path is None:
            args.clef_path = os.path.join(args.clef_dir, f"clef_{args.clef_size}.ckpt")
            print(f"CLEF path auto-set: {args.clef_path}")
        print(f"Loading CLEF encoder ({args.clef_size}) ...")
        clef_encoder = load_clef_encoder(args.clef_path, args.clef_size, device)

    align_tag = "aligned" if args.apply_align else "unaligned"
    run_name = f"{args.loss_type}_{align_tag}_fold{args.fold:02d}"
    ckpt_dir = os.path.join(args.output_dir, run_name, "checkpoints")

    print(f"\n  apply_align={args.apply_align}  loss={args.loss_type}  lambda_clinical={args.lambda_clinical}  "
          f"huber_delta={args.huber_delta}  lr={args.lr}  H={args.hidden_size}  "
          f"batch_size={args.batch_size}  epochs={args.epochs}\n")

    model, history = train_fold(
        fold_idx=args.fold,
        fold_subjects=test_subs,
        train_loader=train_loader,
        val_loader=val_loader,
        clef_encoder=clef_encoder,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        lambda_clinical=args.lambda_clinical,
        huber_delta=args.huber_delta,
        hidden_size=args.hidden_size,
        checkpoint_dir=ckpt_dir,
        use_wandb=False,
        model_name="reheartnet",
        loss_type=args.loss_type,
        lr_schedule="linear_decay",
        early_stop_patience=args.early_stop_patience,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{run_name}_history.json")
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)

    train_metrics = quick_metrics(model, train_eval_loader, device)
    val_metrics = quick_metrics(model, val_loader, device)
    test_metrics = quick_metrics(model, test_loader, device)

    print(f"\n=== Summary ({run_name}) ===")
    print(f"train loss: epoch1={history['train'][0]:.5f}  "
          f"epoch{len(history['train'])}={history['train'][-1]:.5f}  "
          f"min={min(history['train']):.5f}")
    print(f"val   loss: epoch1={history['val'][0]:.5f}  "
          f"epoch{len(history['val'])}={history['val'][-1]:.5f}  "
          f"min={min(history['val']):.5f}")
    print(f"\nbest-checkpoint metrics ({'phase-aligned' if args.apply_align else 'unaligned'} "
          f"train/val, always-unaligned test):")
    for split, m in (("train", train_metrics), ("val  ", val_metrics), ("test ", test_metrics)):
        print(f"  {split}: RMSE={m['rmse']:.4f}  PRD={m['prd']:6.2f}%  r={m['pearson_r']:+.4f}")

    summary = {
        "fold": args.fold, "apply_align": args.apply_align, "epochs": args.epochs,
        "lr": args.lr, "batch_size": args.batch_size, "hidden_size": args.hidden_size,
        "loss_type": args.loss_type,
        "train_metrics": train_metrics, "val_metrics": val_metrics, "test_metrics": test_metrics,
    }
    summary_path = os.path.join(args.output_dir, f"{run_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved loss history to {out_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
