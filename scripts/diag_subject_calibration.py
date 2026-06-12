"""Diagnose whether a small amount of per-subject calibration data closes
the cross-subject generalization gap (see memory:
project_cross_subject_generalization_failure).

The `diag_phase_align_traintest.py` sweep showed train->val r-drop is small
(~0.03-0.04) but val->test r-drop is huge (~0.6-0.7) in ALL conditions --
the model fits training subjects well but does not transfer to unseen
subjects. This script tests a concrete mitigation: simulate a short
per-subject calibration session (e.g. the first ~20% of a new subject's
recording, in temporal order) and fine-tune a copy of an already-trained
checkpoint on just that slice, then re-evaluate on the REMAINING (held-out)
portion of that same subject's recording.

For each of the fold's test subjects:
  1. Build that subject's windows (always apply_align=False, matching test
     preprocessing) and split them CHRONOLOGICALLY into a calibration slice
     (first --calib-frac) and an eval slice (the rest).
  2. Load the starting checkpoint into a fresh model and record BASELINE
     RMSE/PRD/pearson_r on the eval slice (no fine-tuning).
  3. Fine-tune that model on the calibration slice for --calib-epochs at
     --calib-lr (small, to avoid catastrophic forgetting on a tiny set),
     reusing train_one_epoch.
  4. Record CALIBRATED metrics on the same eval slice.

Reports per-subject and aggregate (mean across subjects) baseline vs.
calibrated metrics. If calibration substantially improves eval PRD/r, that's
strong evidence the PPG->ECG mapping is subject-specific and a concrete,
reportable mitigation; if not, the generalization gap is likely something
deeper than a per-subject offset/scale mismatch.

Usage:
    python scripts/diag_subject_calibration.py --fold 6 \\
        --checkpoint results/diag_phase_align_traintest/mse_aligned_fold06/checkpoints/reheartnet_fold_06_best.pt
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

from core.data_loader import build_test_dataset
from core.models.baselines import get_model
from core.train import train_one_epoch


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
    ap.add_argument("--checkpoint", type=str,
                    default="results/diag_phase_align_traintest/mse_aligned_fold06/checkpoints/reheartnet_fold_06_best.pt")
    ap.add_argument("--fold-assignments", type=str,
                    default="results/comparison_arch/fold_assignments.json")
    ap.add_argument("--calib-frac", type=float, default=0.2,
                    help="Fraction of each test subject's windows (in temporal "
                         "order) used as the calibration slice. The remainder "
                         "is the held-out eval slice.")
    ap.add_argument("--calib-epochs", type=int, default=20)
    ap.add_argument("--calib-lr", type=float, default=1e-4)
    ap.add_argument("--loss-type", type=str, default="mse", choices=["mse", "huber"],
                    help="Loss used for the fine-tuning step (should match the "
                         "loss the starting checkpoint was trained with).")
    ap.add_argument("--huber-delta", type=float, default=1.71377251715061)
    ap.add_argument("--hidden-size", type=int, default=32,
                    help="Fallback if the checkpoint doesn't store hidden_size.")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--window-sec", type=float, default=None)
    ap.add_argument("--overlap-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default="results/diag_subject_calibration")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.fold_assignments) as f:
        assignments = json.load(f)
    test_subs = assignments[f"fold_{args.fold:02d}"]["test"]
    print(f"Fold {args.fold}: {len(test_subs)} test subjects: {test_subs}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    hidden_size = ckpt.get("hidden_size", args.hidden_size)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  origin fold={ckpt.get('fold')}  epoch={ckpt.get('epoch')}  "
          f"val_loss={ckpt.get('val_loss')}  hidden_size={hidden_size}")

    if args.loss_type == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.HuberLoss(delta=args.huber_delta)

    print(f"\n  calib_frac={args.calib_frac}  calib_epochs={args.calib_epochs}  "
          f"calib_lr={args.calib_lr}  loss={args.loss_type}\n")

    per_subject = []
    for subj in test_subs:
        ds = build_test_dataset([subj], window_sec=args.window_sec, overlap_frac=args.overlap_frac)
        n = len(ds)
        n_calib = max(1, min(n - 1, int(round(n * args.calib_frac))))
        n_eval = n - n_calib

        calib_ds = Subset(ds, range(0, n_calib))
        eval_ds = Subset(ds, range(n_calib, n))
        calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=True,
                                   num_workers=0, pin_memory=True)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=0, pin_memory=True)

        torch.manual_seed(args.seed)
        model = get_model("reheartnet", hidden_size=hidden_size).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        baseline = quick_metrics(model, eval_loader, device)

        optimizer = optim.Adam(model.parameters(), lr=args.calib_lr)
        for _ in range(args.calib_epochs):
            train_one_epoch(model, calib_loader, criterion, optimizer, device)

        calibrated = quick_metrics(model, eval_loader, device)

        print(f"{subj}: n_windows={n} (calib={n_calib}, eval={n_eval})")
        print(f"  baseline  : RMSE={baseline['rmse']:.4f}  PRD={baseline['prd']:6.2f}%  r={baseline['pearson_r']:+.4f}")
        print(f"  calibrated: RMSE={calibrated['rmse']:.4f}  PRD={calibrated['prd']:6.2f}%  r={calibrated['pearson_r']:+.4f}")

        per_subject.append({
            "subject": subj, "n_windows": n, "n_calib": n_calib, "n_eval": n_eval,
            "baseline": baseline, "calibrated": calibrated,
        })

    def _mean(key_path):
        vals = [s[key_path[0]][key_path[1]] for s in per_subject]
        return float(np.nanmean(vals))

    aggregate = {
        "baseline":   {k: _mean(("baseline", k)) for k in ("rmse", "prd", "pearson_r")},
        "calibrated": {k: _mean(("calibrated", k)) for k in ("rmse", "prd", "pearson_r")},
    }

    print("\n=== Aggregate (mean over test subjects) ===")
    for split, m in (("baseline  ", aggregate["baseline"]), ("calibrated", aggregate["calibrated"])):
        print(f"  {split}: RMSE={m['rmse']:.4f}  PRD={m['prd']:6.2f}%  r={m['pearson_r']:+.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    run_name = f"calib_fold{args.fold:02d}_{args.loss_type}_frac{args.calib_frac:g}"
    summary = {
        "fold": args.fold, "checkpoint": args.checkpoint, "loss_type": args.loss_type,
        "calib_frac": args.calib_frac, "calib_epochs": args.calib_epochs, "calib_lr": args.calib_lr,
        "hidden_size": hidden_size, "per_subject": per_subject, "aggregate": aggregate,
    }
    summary_path = os.path.join(args.output_dir, f"{run_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
