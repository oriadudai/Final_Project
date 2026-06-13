"""Head-to-head comparison of the three loss-ablation models
(reheartnet_original, reheartnet_huber, reheartnet_clef) on a COMMON test
window configuration, with AND without per-subject calibration.

Two threads from this session converge here:
  - [[project_window_size_emd_ks_isolation]]: original/huber's EMD/KS are
    evaluated on 4s/0%-overlap/bandpassed windows, clef's on 10s/50%-overlap/
    no-bandpass -- a window-size confound when comparing them directly.
  - [[project_cross_subject_generalization_failure]]: ~100s of per-subject
    calibration fine-tuning dramatically improves PRD/r/EMD/KS/beat-MAE for a
    held-out subject (confirmed for mse_aligned, fold06).

This script removes BOTH confounds at once: for each of the three trained
checkpoints (results/comparison_reheartnet/{model}/checkpoints/
reheartnet_fold_{NN}_best.pt, for whichever folds exist), and each of that
fold's test subjects:
  1. Build the subject's windows at a COMMON configuration (default: 10s,
     50% overlap, no bandpass -- clef's protocol), apply_align=False.
  2. Split chronologically into a calibration slice (first --calib-frac) and
     an eval slice (the rest).
  3. BASELINE: evaluate the unmodified checkpoint on the eval slice.
  4. CALIBRATE: fine-tune a copy of the checkpoint on the calibration slice
     (--calib-epochs, reusing train_one_epoch). By default (--calib-loss
     auto), models trained with loss_type="clef" (reheartnet_clef,
     arch_reheartnet) are calibrated with the SAME composite objective used
     in training -- Huber(delta=--huber-delta) + --lambda-clinical * CLEF
     feature loss, via ClinicalCompositeLoss -- so calibration doesn't
     fine-tune away the CLEF-learned rhythm structure that gives these models
     their EMD/KS edge (see project_clef_calibration_emd_tradeoff, where
     Huber-only calibration of arch_reheartnet improved PRD/r but degraded
     EMD/KS). mse/huber-trained models default to Huber-only, per a prior
     mse-vs-huber calibration sweep that found near-identical results for
     those. --calib-loss huber/composite overrides this choice for ALL
     --models, e.g. to test whether the CLEF feature loss helps calibration
     even for a model that was never trained with it.
  5. CALIBRATED: re-evaluate on the same eval slice.

Reports, per model, RMSE/PRD/pearson_r/EMD/KS/beat_timing_mae pooled across
all available subjects/folds, baseline vs calibrated -- a fair comparison of
the three loss functions under one test protocol, both as population models
and after personalization.

In addition to the three `compare_reheartnet.MODELS` keys, `--models` also
accepts keys from EXTRA_MODELS below -- checkpoints produced by run_cv.py
(not compare_reheartnet.py), which live at a fixed `checkpoints/` path keyed
only by model_name + fold, independent of --results-dir. This lets e.g.
arch_reheartnet (ReHeartNet+CLEF trained at Optuna's lr, the architectural-
ablation checkpoint) be compared against reheartnet_original/reheartnet_huber
re-trained at the same Optuna lr (see --results-dir override).

Usage:
    python scripts/diag_calib_loss_ablation.py
    python scripts/diag_calib_loss_ablation.py --folds 0,1 --models reheartnet_clef
    python scripts/diag_calib_loss_ablation.py \\
        --results-dir results/comparison_reheartnet_optunalr \\
        --models reheartnet_original,reheartnet_huber,arch_reheartnet
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.config as config
from core.data_loader import build_test_dataset
from core.models.baselines import get_model
from core.train import train_one_epoch
from core.losses.composite_loss import ClinicalCompositeLoss, load_clef_encoder
from compare_reheartnet import MODELS
from diag_subject_calibration import quick_metrics, chronological_calib_eval_split, aggregate_with_ci

AGG_KEYS = ("rmse", "prd", "pearson_r", "emd", "ks_stat", "beat_timing_mae")

# Models trained via run_cv.py rather than compare_reheartnet.py: checkpoints
# land in the single fixed config.CHECKPOINT_DIR ("checkpoints/"), named only
# by model_name + fold -- NOT under {results_dir}/{model_key}/checkpoints/.
EXTRA_MODELS = {
    "arch_reheartnet": {
        "label": "ReHeartNet + CLEF (Optuna lr, arch ablation)",
        "loss_type": "clef",
        "ckpt_path": lambda fold_idx: os.path.join(
            config.CHECKPOINT_DIR, f"reheartnet_fold_{fold_idx:02d}_best.pt"),
        "hidden_size": config.HIDDEN_SIZE,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="reheartnet_original,reheartnet_huber,reheartnet_clef")
    ap.add_argument("--results-dir", type=str, default=os.path.join("results", "comparison_reheartnet"),
                    help="Directory containing {model_key}/checkpoints/ and fold_assignments.json.")
    ap.add_argument("--fold-assignments", type=str, default=None,
                    help="Defaults to <results-dir>/fold_assignments.json")
    ap.add_argument("--folds", type=str, default=None,
                    help="Comma-separated fold indices to run (default: all folds in fold_assignments)")
    ap.add_argument("--window-sec", type=float, default=None,
                    help="Common test window size in seconds (default: None -> config 10s)")
    ap.add_argument("--overlap-frac", type=float, default=0.5)
    ap.add_argument("--apply-bandpass", action="store_true", default=False)
    ap.add_argument("--calib-frac", type=float, default=0.2,
                    help="Fraction of each subject's windows (temporal order) used for calibration.")
    ap.add_argument("--calib-epochs", type=int, default=20)
    ap.add_argument("--calib-lr", type=float, default=1e-4)
    ap.add_argument("--huber-delta", type=float, default=1.71377251715061,
                    help="Delta for the Huber loss used in calibration fine-tuning "
                         "(Huber-only models, and the Huber term of the composite loss).")
    ap.add_argument("--calib-loss", type=str, default="auto",
                    choices=["auto", "huber", "mse", "composite", "clef-only"],
                    help="Calibration objective for ALL --models. 'auto' (default): "
                         "composite for loss_type='clef' models, Huber-only otherwise. "
                         "'huber'/'mse': pure distortion losses (Huber(--huber-delta) "
                         "or plain MSE). 'composite': Huber + lambda*CLEF (the training "
                         "objective for loss_type='clef' models). 'clef-only': pure "
                         "CLEF feature-matching loss (huber_weight=0), the opposite "
                         "extreme from 'huber'/'mse'. Any of these can be forced for "
                         "ALL --models regardless of how they were trained.")
    ap.add_argument("--lambda-clinical", type=float, default=None,
                    help="Weight for the CLEF feature-matching term in the composite "
                         "calibration loss (used whenever composite calibration is "
                         "active, per --calib-loss). Default: read from "
                         "results/best_hyperparams.json, falling back to "
                         "config.LAMBDA_CLINICAL.")
    ap.add_argument("--clef-path", type=str, default=None)
    ap.add_argument("--clef-dir", type=str, default=config.CLEF_CHECKPOINT_DIR)
    ap.add_argument("--clef-size", type=str, default="auto", choices=["auto", "small", "medium", "large"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default=os.path.join("results", "diag_calib_loss_ablation"))
    args = ap.parse_args()

    device = config.DEVICE
    print(f"Device: {device}")
    print(f"Common test windows: window_sec={args.window_sec or config.WINDOW_SIZE}  "
          f"overlap={args.overlap_frac:.0%}  bandpass={args.apply_bandpass}")
    print(f"calib_frac={args.calib_frac}  calib_epochs={args.calib_epochs}  "
          f"calib_lr={args.calib_lr}  huber_delta={args.huber_delta}  "
          f"calib_loss={args.calib_loss} (per-model resolution -- see below)")

    fold_assignments_path = args.fold_assignments or os.path.join(args.results_dir, "fold_assignments.json")
    with open(fold_assignments_path) as f:
        assignments = json.load(f)
    fold_keys = sorted(assignments.keys())
    if args.folds:
        wanted = {int(x) for x in args.folds.split(",")}
        fold_keys = [k for k in fold_keys if int(k.split("_")[1]) in wanted]
    print(f"Folds: {fold_keys}")

    model_keys = [k.strip() for k in args.models.split(",")]

    def _loss_type(model_key):
        cfg = MODELS.get(model_key) or EXTRA_MODELS.get(model_key) or {}
        return cfg.get("loss_type")

    def _calib_loss_kind(model_key):
        if args.calib_loss == "auto":
            return "composite" if _loss_type(model_key) == "clef" else "huber"
        return args.calib_loss

    huber_criterion = nn.HuberLoss(delta=args.huber_delta)
    mse_criterion = nn.MSELoss()
    clef_criterion = None
    clef_only_criterion = None
    needed_kinds = {_calib_loss_kind(k) for k in model_keys}
    if needed_kinds & {"composite", "clef-only"}:
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
        print(f"Loading CLEF encoder ({args.clef_size}) for composite/clef-only calibration ...")
        clef_encoder = load_clef_encoder(args.clef_path, args.clef_size, device)
        if "composite" in needed_kinds:
            clef_criterion = ClinicalCompositeLoss(
                clef_encoder, args.lambda_clinical, args.huber_delta, huber_weight=1.0).to(device)
        if "clef-only" in needed_kinds:
            clef_only_criterion = ClinicalCompositeLoss(
                clef_encoder, args.lambda_clinical, args.huber_delta, huber_weight=0.0).to(device)

    results = {}
    for model_key in model_keys:
        if model_key in MODELS:
            model_cfg = MODELS[model_key]
            label = model_cfg["label"]
            default_hidden_size = model_cfg.get("hidden_size", config.HIDDEN_SIZE)
            ckpt_path_fn = lambda fold_idx, _mk=model_key: os.path.join(
                args.results_dir, _mk, "checkpoints", f"reheartnet_fold_{fold_idx:02d}_best.pt")
        elif model_key in EXTRA_MODELS:
            model_cfg = EXTRA_MODELS[model_key]
            label = model_cfg["label"]
            default_hidden_size = model_cfg.get("hidden_size", config.HIDDEN_SIZE)
            ckpt_path_fn = model_cfg["ckpt_path"]
        else:
            print(f"\nSkipping unknown model key: {model_key} "
                  f"(not in compare_reheartnet.MODELS or EXTRA_MODELS)")
            continue
        print(f"\n{'-'*60}\n  Model: {label}\n{'-'*60}")

        calib_loss_kind = _calib_loss_kind(model_key)
        auto_kind = "composite" if model_cfg.get("loss_type") == "clef" else "huber"
        if args.calib_loss != "auto" and calib_loss_kind != auto_kind:
            forced_note = (f"  [non-default for this model; loss_type="
                            f"{model_cfg.get('loss_type')!r}, auto would use {auto_kind!r}]")
        else:
            forced_note = ""

        if calib_loss_kind == "composite":
            criterion = clef_criterion
            print(f"  Calibration loss: Huber(delta={args.huber_delta}) + "
                  f"{args.lambda_clinical} * CLEF feature loss{forced_note}")
        elif calib_loss_kind == "clef-only":
            criterion = clef_only_criterion
            print(f"  Calibration loss: {args.lambda_clinical} * CLEF feature loss only "
                  f"(huber_weight=0){forced_note}")
        elif calib_loss_kind == "mse":
            criterion = mse_criterion
            print(f"  Calibration loss: MSE{forced_note}")
        else:
            criterion = huber_criterion
            print(f"  Calibration loss: Huber(delta={args.huber_delta}){forced_note}")

        per_subject = []
        for fold_key in fold_keys:
            fold_idx = int(fold_key.split("_")[1])
            test_subs = assignments[fold_key]["test"]

            ckpt_path = ckpt_path_fn(fold_idx)
            if not os.path.exists(ckpt_path):
                print(f"  fold {fold_idx:02d}: checkpoint not found ({ckpt_path}), skipping.")
                continue

            ckpt = torch.load(ckpt_path, map_location=device)
            hidden_size = ckpt.get("hidden_size", default_hidden_size)

            for subj in test_subs:
                ds = build_test_dataset([subj], window_sec=args.window_sec,
                                         overlap_frac=args.overlap_frac,
                                         apply_bandpass=args.apply_bandpass)
                n = len(ds)
                calib_ds, eval_ds, n_calib, n_eval = chronological_calib_eval_split(
                    ds, args.calib_frac, args.overlap_frac)
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

                print(f"    {subj} (fold{fold_idx:02d}, n={n}): "
                      f"RMSE {baseline['rmse']:.3f}->{calibrated['rmse']:.3f}  "
                      f"PRD {baseline['prd']:6.2f}%->{calibrated['prd']:6.2f}%  "
                      f"r {baseline['pearson_r']:+.3f}->{calibrated['pearson_r']:+.3f}  "
                      f"EMD {baseline['emd']:.3f}->{calibrated['emd']:.3f}  "
                      f"KS {baseline['ks_stat']:.3f}->{calibrated['ks_stat']:.3f}  "
                      f"beat-MAE {baseline['beat_timing_mae']:.3f}->{calibrated['beat_timing_mae']:.3f}")

                per_subject.append({
                    "fold": fold_idx, "subject": subj, "n_windows": n,
                    "n_calib": n_calib, "n_eval": n_eval,
                    "baseline": baseline, "calibrated": calibrated,
                })

        if not per_subject:
            print(f"  No checkpoints found for {model_key}, skipping.")
            continue

        aggregate = aggregate_with_ci(per_subject, AGG_KEYS)

        n_folds_used = len(set(s["fold"] for s in per_subject))
        print(f"\n  === {model_key}: {len(per_subject)} subjects across {n_folds_used} fold(s) ===")
        for split, m in (("baseline  ", aggregate["baseline"]), ("calibrated", aggregate["calibrated"])):
            print(f"    {split}: RMSE={m['rmse']['mean']:.4f}+/-{m['rmse']['ci95_margin']:.4f}  "
                  f"PRD={m['prd']['mean']:6.2f}+/-{m['prd']['ci95_margin']:.2f}%  "
                  f"r={m['pearson_r']['mean']:+.4f}+/-{m['pearson_r']['ci95_margin']:.4f}  "
                  f"EMD={m['emd']['mean']:.4f}+/-{m['emd']['ci95_margin']:.4f}  "
                  f"KS={m['ks_stat']['mean']:.3f}+/-{m['ks_stat']['ci95_margin']:.3f}  "
                  f"beat-MAE={m['beat_timing_mae']['mean']:.4f}+/-{m['beat_timing_mae']['ci95_margin']:.4f}s")

        results[model_key] = {
            "label": label, "n_subjects": len(per_subject), "n_folds": n_folds_used,
            "per_subject": per_subject, "aggregate": aggregate,
        }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "calib_loss_ablation_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
