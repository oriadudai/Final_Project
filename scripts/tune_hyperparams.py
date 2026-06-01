"""Optuna hyperparameter search for ReHeartNet.

Runs a 2-fold proxy CV on a small subject subset to find good hyperparameters
before the full 8-fold CV.

Runtime guide (CPU):
  --fast   : 5 subjects, 5 epochs, 10 trials  → ~15-20 min  (recommended first run)
  default  : 6 subjects, 10 epochs, 20 trials → ~45-60 min
  --full   : 14 subjects, 20 epochs, 50 trials → ~4-8 h

Usage:
    python scripts/tune_hyperparams.py --clef-path models/clef/clef_small.ckpt --fast
    python scripts/tune_hyperparams.py --clef-path models/clef/clef_small.ckpt
    python scripts/tune_hyperparams.py --clef-path models/clef/clef_small.ckpt --full
    # Resume a previous study:
    python scripts/tune_hyperparams.py --clef-path models/clef/clef_small.ckpt --resume
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

# Ensure project root is on sys.path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.config as config
from core.data_loader import build_group_fold, split_train_val
from core.losses.composite_loss import ClinicalCompositeLoss, load_clef_encoder
from core.models.ptbxl_classifier import get_ptbxl_classifier
from core.train import train_fold
from core.evaluate import evaluate_fold

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError as e:
    raise SystemExit("optuna is required: pip install optuna plotly kaleido") from e

# ---------------------------------------------------------------------------
# Global (built once, shared across all trials to avoid redundant loading)
# ---------------------------------------------------------------------------
DEVICE: torch.device = config.DEVICE
GLOBAL_CLEF_ENCODER = None
GLOBAL_PTBXL_CLF    = None

# Subject counts per mode — set in main() after parsing args
TUNE_SUBJECTS: list = []
TUNE_SPLITS:   list = []


def objective(trial: "optuna.Trial", epochs: int) -> float:
    lr          = trial.suggest_float("lr",               1e-4, 1e-2, log=True)
    lambda_clin = trial.suggest_float("lambda_clinical",  1e-3, 1.0,  log=True)
    huber_delta = trial.suggest_float("huber_delta",      0.1,  2.0)
    batch_size  = trial.suggest_categorical("batch_size", [32, 64, 128])
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])

    fold_prds = []

    for fold_idx, (train_idx, test_idx) in enumerate(TUNE_SPLITS):
        train_subs = [TUNE_SUBJECTS[i] for i in train_idx]
        test_subs  = [TUNE_SUBJECTS[i] for i in test_idx]

        train_ds, test_ds = build_group_fold(train_subs, test_subs, apply_align=True)
        train_inner, val_ds = split_train_val(train_ds, val_fraction=0.15)

        train_loader = DataLoader(train_inner, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,      batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,     batch_size=batch_size, shuffle=False, num_workers=0)

        model, _ = train_fold(
            fold_idx       = fold_idx,
            fold_subjects  = test_subs,
            train_loader   = train_loader,
            val_loader     = val_loader,
            clef_encoder   = GLOBAL_CLEF_ENCODER,
            device         = DEVICE,
            epochs         = epochs,
            lr             = lr,
            lambda_clinical = lambda_clin,
            huber_delta    = huber_delta,
            hidden_size    = hidden_size,
            checkpoint_dir = os.path.join("results", "tune_checkpoints"),
            use_wandb      = False,
        )

        metrics = evaluate_fold(model, test_loader, GLOBAL_PTBXL_CLF, DEVICE)
        prd = metrics.get("prd", float("nan"))
        if not np.isnan(prd):
            fold_prds.append(prd)

        # Optuna pruning: report intermediate result and check if trial should stop
        mean_so_far = float(np.mean(fold_prds)) if fold_prds else float("inf")
        trial.report(mean_so_far, step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_prds)) if fold_prds else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for ReHeartNet")
    parser.add_argument("--clef-path",        type=str, required=True, help="Path to CLEF .ckpt file")
    parser.add_argument("--clef-size",        type=str, default="small", choices=["small","medium","large"])
    parser.add_argument("--classifier-path",  type=str, default=None, help="Path to PTB-XL classifier .pt")
    parser.add_argument("--n-trials",         type=int, default=None, help="Override trial count")
    parser.add_argument("--epochs",           type=int, default=None, help="Override epochs per trial")
    parser.add_argument("--timeout-hours",    type=float, default=None, help="Override time budget")
    parser.add_argument("--resume",           action="store_true", help="Resume existing Optuna study")
    # Speed presets
    speed = parser.add_mutually_exclusive_group()
    speed.add_argument("--fast", action="store_true",
                       help="5 subjects, 5 epochs, 10 trials (~15-20 min on CPU)")
    speed.add_argument("--full", action="store_true",
                       help="14 subjects, 20 epochs, 50 trials (~4-8 h on CPU)")
    parser.add_argument("--output-dir",       type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    # Apply speed preset (--fast / default / --full)
    global TUNE_SUBJECTS, TUNE_SPLITS
    if args.fast:
        n_subj, epochs, n_trials, timeout_h = 5, 5, 10, 1.0
        print("Mode: --fast  (5 subjects, 5 epochs, 10 trials, ~15-20 min on CPU)")
    elif args.full:
        n_subj, epochs, n_trials, timeout_h = 14, 20, 50, 8.0
        print("Mode: --full  (14 subjects, 20 epochs, 50 trials, ~4-8 h on CPU)")
    else:
        n_subj, epochs, n_trials, timeout_h = 6, 10, 20, 2.0
        print("Mode: default (6 subjects, 10 epochs, 20 trials, ~45-60 min on CPU)")

    # CLI overrides take precedence over preset values
    if args.epochs        is not None: epochs    = args.epochs
    if args.n_trials      is not None: n_trials  = args.n_trials
    if args.timeout_hours is not None: timeout_h = args.timeout_hours

    TUNE_SUBJECTS = [f"bidmc{i:02d}" for i in range(1, n_subj + 1)]
    TUNE_SPLITS   = list(KFold(n_splits=2, shuffle=True, random_state=42).split(TUNE_SUBJECTS))
    print(f"Subjects: {TUNE_SUBJECTS}  |  epochs/trial: {epochs}  |  trials: {n_trials}  |  budget: {timeout_h}h\n")

    # Build shared models once
    global GLOBAL_CLEF_ENCODER, GLOBAL_PTBXL_CLF
    print(f"Loading CLEF encoder ({args.clef_size}) from {args.clef_path} ...")
    GLOBAL_CLEF_ENCODER = load_clef_encoder(args.clef_path, args.clef_size, DEVICE)
    GLOBAL_PTBXL_CLF    = get_ptbxl_classifier(args.classifier_path).to(DEVICE)

    db_path    = os.path.join(args.output_dir, "optuna_study.db")
    study_name = "reheartnet_tuning"

    study = optuna.create_study(
        study_name    = study_name,
        direction     = "minimize",
        sampler       = TPESampler(seed=42),
        pruner        = MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        storage       = f"sqlite:///{db_path}",
        load_if_exists= True,   # always allow resume via --resume or fresh run
    )

    study.optimize(
        lambda trial: objective(trial, epochs),
        n_trials  = n_trials,
        timeout   = timeout_h * 3600,
        show_progress_bar=True,
    )

    # Save best hyperparameters
    best_params = study.best_params
    best_path   = os.path.join(args.output_dir, "best_hyperparams.json")
    with open(best_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to {best_path}:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"  Best PRD: {study.best_value:.4f}")

    # Optuna visualizations (requires plotly + kaleido)
    try:
        import plotly  # noqa: F401
        fig_hist = optuna.visualization.plot_optimization_history(study)
        fig_hist.write_image(os.path.join(args.output_dir, "figures", "optuna_history.png"))
        fig_imp  = optuna.visualization.plot_param_importances(study)
        fig_imp.write_image(os.path.join(args.output_dir, "figures", "optuna_importances.png"))
        print("Optuna visualizations saved.")
    except Exception as e:
        print(f"Could not save Optuna plots (plotly/kaleido may be missing): {e}")


if __name__ == "__main__":
    main()
