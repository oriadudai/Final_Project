"""Three-way comparison: original ReHeartNet vs. our loss-function improvements.

Three model variants, all using the same DC-BiLSTM architecture and the same
training protocol as Lee et al. (2026) — batch=1, lr=1e-2, ×0.75 linear decay
every 50 epochs, 1000 epochs, no early stopping.  Only the loss function changes:

  1. reheartnet_original -- faithful replica of Lee et al.:
       MSE loss, 4 s windows, FIR bandpass, paper training protocol.

  2. reheartnet_huber    -- same protocol, Huber loss.
       Optuna tunes huber_delta and hidden_size.

  3. reheartnet_clef     -- same protocol, Huber + CLEF composite loss.
       10 s windows (CLEF encoder requires 10 s input).
       Optuna tunes lambda_clinical, huber_delta, and hidden_size.

Outputs (organised per experiment — each variant's weights, metrics, and
per-fold figures live together under its own directory):
  results/comparison_reheartnet/{model}/checkpoints/   per-fold model weights
  results/comparison_reheartnet/{model}/figures/       per-fold loss/recon plots
  results/comparison_reheartnet/{model}/partial.json   incremental per-fold metrics (--resume)
  results/comparison_reheartnet/{model}/summary.json   per-model summary (mean +/- CI)
  results/comparison_reheartnet/comparison.json        cross-model side-by-side table
  results/comparison_reheartnet/figures/               cross-model comparison plots

Usage:
    python scripts/compare_reheartnet.py --clef-path models/clef/clef_small.ckpt
    python scripts/compare_reheartnet.py --clef-path models/clef/clef_small.ckpt --dry-run
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.config as config
from core.data_loader import build_group_fold, build_test_dataset, get_cv_splits, split_train_val
from core.metrics.clinical_metrics import compute_bce
from core.models.ptbxl_classifier import build_classifier
from core.evaluate import evaluate_fold
from core.losses.composite_loss import load_clef_encoder
from core.metrics.clinical_metrics import extract_rr_intervals
from core.models.ptbxl_classifier import get_ptbxl_classifier
from core.train import train_fold
from core.visualization.plots import (
    plot_loss_curves,
    plot_reconstruction_samples,
    plot_rr_distributions,
    save_results_summary,
)
from src.preprocessing import get_all_record_names

# ------------------------------------------------------------------------------
# Three model variants -- same DC-BiLSTM architecture, same training protocol
# as Lee et al. (2026), differing only in loss function.
#
#  original : MSE                   <- paper baseline
#  huber    : Huber                 <- our contribution 1
#  clef     : Huber + CLEF          <- our full method
#
# All variants:
#   - 8-fold subject-grouped CV (same splits, same random seed)
#   - Training protocol from Lee et al.: batch=1, lr=1e-2, ×0.75/50ep, 1000ep
#   - Unspecified paper values (hidden_size, loss HPs) -> Optuna / config default
#
# Keys in each entry:
#   loss_type          : "mse" | "huber" | "clef"
#   window_sec         : float (seconds) or None -> config default (10 s)
#   overlap_frac       : 0.0 = non-overlapping, 0.5 = 50% overlap
#   apply_bandpass     : True -> FIR ECG 0.5-55 Hz / PPG 0.5-10 Hz (Lee et al.)
#   batch_size / epochs / lr / lr_schedule / early_stop_patience: training protocol
#   use_optuna         : bool -- if False, best_params never consulted;
#                        unspecified values fall back to config defaults
# ------------------------------------------------------------------------------
MODELS = {
    "reheartnet_original": {
        "label":               "ReHeartNet (original)",
        "loss_type":           "mse",
        # preprocessing -- matches Lee et al. supplementary exactly
        "window_sec":          4.0,
        "overlap_frac":        0.0,
        "apply_bandpass":      True,
        # training -- matches Lee et al. supplementary exactly
        "batch_size":          1,
        "epochs":              1000,
        "lr":                  1e-2,
        # Paper does not specify hidden_size; pinned to the Optuna-found value
        # (32) so all three loss variants share one hidden_size, keeping the
        # loss function the only variable that differs between them.
        "hidden_size":         32,
        "lr_schedule":         "linear_decay",
        "early_stop_patience": None,          # run all 1000 epochs
        # Paper did not use Optuna; anything not above falls back to config defaults
        "use_optuna":          False,
    },
    "reheartnet_huber": {
        "label":               "ReHeartNet + Huber (ours)",
        "loss_type":           "huber",
        # Same preprocessing and training protocol as the original paper, AND
        # pinned to the same lr/hidden_size as reheartnet_original -- so this
        # is a true apples-to-apples test of MSE vs. Huber, isolating the loss
        # function as the only variable. huber_delta is left Optuna-tuned since
        # it's intrinsic to the loss being evaluated (no paper baseline exists).
        "window_sec":          4.0,
        "overlap_frac":        0.0,
        "apply_bandpass":      True,
        "batch_size":          1,
        "epochs":              1000,
        "lr":                  1e-2,
        "hidden_size":         32,
        "lr_schedule":         "linear_decay",
        "early_stop_patience": None,
    },
    "reheartnet_clef": {
        "label":               "ReHeartNet + Huber + CLEF (ours)",
        "loss_type":           "clef",
        # 10 s window required: CLEF encoder expects 10 s input (5000 samples at
        # 500 Hz). Feeding it 4 s would distort temporal content and corrupt BCE.
        # All other training settings match the paper.
        # Optuna tunes loss hyperparameters (lambda_clinical, huber_delta) and hidden_size.
        "window_sec":          None,           # config default = 10 s = 1250 samples
        "overlap_frac":        0.5,
        "apply_bandpass":      False,
        "batch_size":          1,
        "epochs":              1000,
        "lr":                  1e-2,
        "lr_schedule":         "linear_decay",
        "early_stop_patience": None,
    },
}

METRICS = ["rmse", "prd", "pearson_r", "bce", "emd", "ks_stat", "beat_timing_mae"]
METRIC_LABELS = {
    "rmse":            "RMSE (norm.)",
    "prd":             "PRD (%)",
    "pearson_r":       "Pearson r",
    "bce":             "BCE",
    "emd":             "EMD (s)",
    "ks_stat":         "KS statistic",
    "beat_timing_mae": "Beat MAE (s)",
}
# For each metric: True = lower is better, False = higher is better
LOWER_BETTER = {
    "rmse": True, "prd": True, "pearson_r": False, "bce": True,
    "emd": True, "ks_stat": True, "beat_timing_mae": True,
}


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare original ReHeartNet vs. CLEF-enhanced ReHeartNet")
    p.add_argument("--clef-path",       type=str, default=None,
                   help="Path to CLEF .ckpt. If omitted, auto-constructs from --clef-dir and --clef-size.")
    p.add_argument("--clef-size",       type=str, default="auto",  choices=["auto", "small", "medium", "large"])
    p.add_argument("--clef-dir",        type=str, default=config.CLEF_CHECKPOINT_DIR,
                   help="Directory containing clef_*.ckpt files (used when --clef-path is not given).")
    p.add_argument("--classifier-path", type=str, default=None)
    p.add_argument("--output-dir",      type=str, default=os.path.join("results", "comparison_reheartnet"))
    p.add_argument("--n-folds",         type=int, default=8)
    p.add_argument("--epochs",          type=int, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None,
                   help="Stop a fold early after N epochs without val-loss improvement (paper default: None = run all epochs)")
    p.add_argument("--batch-size",      type=int, default=None)
    p.add_argument("--resume",          action="store_true", help="Resume from existing partial results")
    p.add_argument("--only",            type=str, default=None,
                   help="Comma-separated subset of model keys to run (e.g. 'reheartnet_clef'). "
                        "Lets each loss variant be launched as its own process on its own GPU; "
                        "rerun without --only (with --resume) afterward to assemble the comparison.")
    p.add_argument("--dry-run",         action="store_true", help="2 folds, 2 epochs each")
    p.add_argument("--no-wandb",        action="store_true")
    p.add_argument("--wandb-project",   type=str, default="ppg2ecg-reheartnet",
                   help="W&B project name")
    p.add_argument("--wandb-entity",    type=str, default=None,
                   help="W&B entity (username or team). Defaults to your logged-in account.")
    return p.parse_args()


# ------------------------------------------------------------------------------
# Single-model CV loop
# ------------------------------------------------------------------------------

def _run_model(
    model_key:    str,
    model_cfg:    dict,
    args:         argparse.Namespace,
    device:       torch.device,
    clef_encoder: torch.nn.Module,
    ptbxl_clf:    torch.nn.Module,
    splits:       list,
    best_params:  dict,
) -> tuple:
    """Run CV for one model variant.

    Returns (fold_metrics, sample) where sample is a dict with keys
    "true" and "pred" (numpy arrays, up to 50 windows from fold 0) used for
    the multi-model waveform overlay and RR KDE comparison figures.

    Hyperparameter priority (highest -> lowest):
      args CLI override  >  model_cfg explicit value  >  best_params (Optuna)  >  config default

    CLI overrides (--epochs, --early-stop-patience, --batch-size) take top priority
    so that a long paper-spec run (epochs=1000, no early stopping) can be deliberately
    scaled down for tractability without editing MODELS.

    For reheartnet_original, use_optuna=False so best_params is never consulted —
    only paper-specified values (in model_cfg) or config defaults are used.
    """
    # The original paper fixed hyperparameters manually; never pull Optuna values for it.
    hp = best_params if model_cfg.get("use_optuna", True) else {}

    epochs      = args.epochs     or model_cfg.get("epochs")     or int(hp.get("epochs",           config.EPOCHS))
    batch_size  = args.batch_size or model_cfg.get("batch_size") or int(hp.get("batch_size",       config.BATCH_SIZE))
    hidden_size = model_cfg.get("hidden_size") or                   int(hp.get("hidden_size",      config.HIDDEN_SIZE))
    lr          = model_cfg.get("lr")         or                  float(hp.get("lr",               config.LEARNING_RATE))
    huber_delta =                                                  float(hp.get("huber_delta",      config.HUBER_DELTA))
    lambda_clinical =                                              float(hp.get("lambda_clinical",  config.LAMBDA_CLINICAL))

    loss_type           = model_cfg["loss_type"]
    window_sec          = model_cfg.get("window_sec")           # None -> config default
    overlap_frac        = model_cfg.get("overlap_frac", 0.5)
    apply_bandpass      = model_cfg.get("apply_bandpass", False)
    lr_schedule         = model_cfg.get("lr_schedule", "linear_decay")
    early_stop_patience = args.early_stop_patience if args.early_stop_patience is not None \
                          else model_cfg.get("early_stop_patience")

    if args.dry_run:
        epochs = 5         # enough to see emerging metric trends without being too slow
        # Dry-run overrides: pipeline correctness check, not faithful hyperparameters.
        batch_size = max(batch_size, 32)   # batch=1 makes CPU dry-runs impractically slow

    label = model_cfg["label"]
    n_folds = len(splits)
    print(f"\n  loss={loss_type}  |  lr={lr}  |  H={hidden_size}  |  epochs={epochs}  |  folds={n_folds}")
    print(f"  huber_delta={huber_delta}  |  lambda_clinical={lambda_clinical}")
    if window_sec is not None:
        print(f"  window={window_sec}s  overlap={overlap_frac:.0%}  bandpass={apply_bandpass}")

    fold0_sample: dict = {}   # populated on fold 0 for comparison figures

    # Everything for this experiment (weights, partial/summary results, per-fold
    # figures) lives together under one directory, keyed by model_key.
    run_dir = os.path.join(args.output_dir, model_key)
    partial_path = os.path.join(run_dir, "partial.json")
    fold_metrics: list = []
    resume_from = 0
    if args.resume and os.path.exists(partial_path):
        with open(partial_path) as f:
            fold_metrics = json.load(f)
        resume_from = len(fold_metrics)
        print(f"  Resuming {label} from fold {resume_from}.")

    for fold_idx, (train_subs, test_subs) in enumerate(
        tqdm(splits, desc=label, unit="fold")
    ):
        if fold_idx < resume_from:
            continue

        # Limit subjects per fold: 5 subjects × 120 windows × 5 epochs ≈ 3000 steps/fold
        # ~3-4 min for original/Huber, ~6-8 min for CLEF — ~12-15 min total on CPU
        _train = train_subs[:5] if args.dry_run else train_subs
        _test  = test_subs[:2]  if args.dry_run else test_subs
        train_ds, test_ds = build_group_fold(
            _train, _test, apply_align=True,
            window_sec=window_sec, overlap_frac=overlap_frac,
            apply_bandpass=apply_bandpass,
        )
        train_inner, val_ds = split_train_val(train_ds, val_fraction=0.1)

        train_loader = DataLoader(train_inner, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,      batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        test_loader  = DataLoader(test_ds,     batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        model, history = train_fold(
            fold_idx            = fold_idx,
            fold_subjects       = test_subs,
            train_loader        = train_loader,
            val_loader          = val_loader,
            clef_encoder        = clef_encoder,
            device              = device,
            epochs              = epochs,
            lr                  = lr,
            lambda_clinical     = lambda_clinical,
            huber_delta         = huber_delta,
            hidden_size         = hidden_size,
            checkpoint_dir      = os.path.join(run_dir, "checkpoints"),
            use_wandb           = not args.no_wandb,
            wandb_kwargs        = {"project": args.wandb_project,
                                   "entity":  args.wandb_entity,
                                   "group":   model_key},   # groups all folds per model variant
            model_name          = "reheartnet",   # same DC-BiLSTM architecture for all variants
            loss_type           = loss_type,
            lr_schedule         = lr_schedule,
            early_stop_patience = early_stop_patience,
        )

        if window_sec is not None:
            # 4 s model: evaluate all metrics except BCE on the 4 s test windows,
            # then compute BCE separately on 10 s windows (CLEFClassifier needs 10 s input).
            # The BiLSTM is sequence-length agnostic so the trained model runs on 10 s fine.
            metrics = evaluate_fold(model, test_loader, ptbxl_clf, device, clef_encoder=None)

            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    _wandb.log({f"eval/{k}": v for k, v in metrics.items()
                                if isinstance(v, float) and k != "ks_pvalue"})
            except Exception:
                pass

            bce_ds = build_test_dataset(
                _test, window_sec=None, overlap_frac=0.5, apply_bandpass=False,
            )
            bce_loader = DataLoader(bce_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=0, pin_memory=True)
            model.eval()
            true_10s, pred_10s = [], []
            with torch.no_grad():
                for ppg_b, ecg_b in bce_loader:
                    pred_b = model(ppg_b.to(device))
                    true_10s.append(ecg_b.squeeze(-1).cpu().numpy())
                    pred_10s.append(pred_b.squeeze(-1).cpu().numpy())
            true_10s = np.concatenate(true_10s)
            pred_10s = np.concatenate(pred_10s)
            clf = build_classifier(clef_encoder).to(device)
            metrics["bce"] = compute_bce(true_10s, pred_10s, clf, device)
        else:
            # CLEF model: already uses 10 s windows — standard evaluation path.
            metrics = evaluate_fold(model, test_loader, ptbxl_clf, device, clef_encoder=clef_encoder)
            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    _wandb.log({f"eval/{k}": v for k, v in metrics.items()
                                if isinstance(v, float) and k != "ks_pvalue"})
            except Exception:
                pass

        metrics.update({"fold": fold_idx, "subjects": test_subs, "model": model_key})
        fold_metrics.append(metrics)

        # Collect sample predictions from fold 0 for multi-model comparison figures
        if fold_idx == 0 and not fold0_sample:
            model.eval()
            _true_all, _pred_all = [], []
            with torch.no_grad():
                for ppg_b, ecg_b in test_loader:
                    _pred_all.append(model(ppg_b.to(device)).squeeze(-1).cpu().numpy())
                    _true_all.append(ecg_b.squeeze(-1).cpu().numpy())
            fold0_sample = {
                "true":       np.concatenate(_true_all)[:50],
                "pred":       np.concatenate(_pred_all)[:50],
                "window_sec": window_sec,
            }

        print(
            f"    [{label}] fold {fold_idx:02d} -- "
            f"RMSE={metrics['rmse']:.4f}  PRD={metrics['prd']:.3f}  r={metrics['pearson_r']:.3f}  "
            f"BCE={metrics['bce']:.4f}  EMD={metrics['emd']:.4f}  "
            f"KS={metrics['ks_stat']:.3f}  beat-MAE={metrics['beat_timing_mae']:.4f}s"
        )

        # Per-fold figures — kept alongside this experiment's weights and results
        run_fig_dir = os.path.join(run_dir, "figures")
        plot_loss_curves(
            history["train"], history["val"], fold_idx,
            save_path=os.path.join(run_fig_dir, f"loss_fold{fold_idx:02d}.png"),
        )
        _save_recon_figure(model, test_loader, test_subs, device,
                           fold_idx, run_fig_dir)

        with open(partial_path, "w") as f:
            json.dump(fold_metrics, f, indent=2, default=str)

    save_results_summary(
        fold_metrics,
        save_path=os.path.join(run_dir, "summary.json"),
    )
    return fold_metrics, fold0_sample


# ------------------------------------------------------------------------------
# Comparison output
# ------------------------------------------------------------------------------

def _compute_mean_ci(values: list) -> tuple:
    """(mean, ci_margin) for a list of fold values."""
    arr = np.array([v for v in values if not np.isnan(v)])
    if len(arr) == 0:
        return float("nan"), 0.0
    mean   = float(np.mean(arr))
    margin = 1.96 * float(np.std(arr, ddof=1)) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, margin


def _save_comparison_json(
    all_results: dict,  # {model_key: [fold_metric_dicts]}
    output_dir: str,
) -> dict:
    """Build and save side-by-side summary. Returns the summary dict."""
    summary = {}
    for model_key, fold_metrics in all_results.items():
        summary[model_key] = {"label": MODELS[model_key]["label"]}
        for metric in METRICS:
            vals = [m.get(metric, float("nan")) for m in fold_metrics]
            mean, margin = _compute_mean_ci(vals)
            summary[model_key][metric] = {
                "mean":      mean,
                "ci_margin": margin,
                "per_fold":  vals,
            }

    path = os.path.join(output_dir, "comparison.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nComparison saved -> {path}")
    return summary


def _log_wandb_comparison(summary: dict, all_results: dict, args) -> None:
    """Log the final comparison table + per-fold box data as a wandb summary run.

    Creates one final wandb run called 'comparison_summary' in the same project.
    Contains:
      - A wandb.Table with mean +/- CI for all metrics across all models
      - wandb bar charts (logged as custom charts)
      - Per-fold metric values for violin/box plots in wandb
    """
    try:
        import wandb as _wandb  # noqa: PLC0415
    except ImportError:
        return
    if args.no_wandb:
        return

    _wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name="comparison_summary",
        group="summary",
        reinit=True,
    )

    models  = list(summary.keys())
    metrics = [m for m in METRICS if m != "ks_stat"]   # ks_stat less interpretable as bar

    # ── Summary Table ──────────────────────────────────────────────────────────
    cols = ["Model"] + [f"{METRIC_LABELS[m]} (mean)" for m in metrics] \
                     + [f"{METRIC_LABELS[m]} (CI)"   for m in metrics]
    table = _wandb.Table(columns=cols)
    for mk in models:
        row = [summary[mk]["label"]]
        row += [round(summary[mk][m]["mean"],      4) for m in metrics]
        row += [round(summary[mk][m]["ci_margin"],  4) for m in metrics]
        table.add_data(*row)
    _wandb.log({"comparison/summary_table": table})

    # ── Per-fold values (enables violin / box plots in wandb) ─────────────────
    fold_table = _wandb.Table(columns=["model", "fold"] + metrics)
    for mk in models:
        for fm in all_results[mk]:
            row = [summary[mk]["label"], fm.get("fold", "?")]
            row += [round(fm.get(m, float("nan")), 4) for m in metrics]
            fold_table.add_data(*row)
    _wandb.log({"comparison/per_fold_table": fold_table})

    # ── Scalar summary bars (one value per model per metric) ──────────────────
    for m in metrics:
        for mk in models:
            _wandb.summary[f"{summary[mk]['label']}/{METRIC_LABELS[m]}"] = \
                round(summary[mk][m]["mean"], 4)

    _wandb.finish()
    print(f"  Logged comparison summary to wandb project '{args.wandb_project}'")


def _save_results_report(summary: dict, all_results: dict, output_dir: str) -> None:
    """Save a human-readable results report (TXT + per-fold JSON).

    Writes two files:
      results_report.txt  — the comparison table + per-fold breakdown, plain text
      per_fold_results.json — all per-fold values in a flat, easy-to-read structure
    """
    models  = list(summary.keys())
    col_w   = 26

    lines = []
    lines.append("=" * 100)
    lines.append("ReHeartNet Comparison Results")
    lines.append("=" * 100)
    lines.append("")

    # ── Summary table ──────────────────────────────────────────────────────────
    lines.append("SUMMARY  (mean +/- 95% CI)")
    lines.append("-" * 100)
    header = f"{'Metric':<22}"
    for mk in models:
        header += f"  {summary[mk]['label']:^{col_w}}"
    lines.append(header)
    lines.append("-" * 100)

    for metric in METRICS:
        row = f"{METRIC_LABELS[metric]:<22}"
        best_mean = None
        for mk in models:
            m = summary[mk][metric]["mean"]
            if not np.isnan(m):
                if best_mean is None:
                    best_mean = m
                elif LOWER_BETTER[metric] and m < best_mean:
                    best_mean = m
                elif not LOWER_BETTER[metric] and m > best_mean:
                    best_mean = m
        for mk in models:
            m  = summary[mk][metric]["mean"]
            ci = summary[mk][metric]["ci_margin"]
            if np.isnan(m):
                cell = "N/A"
            else:
                marker = "**" if m == best_mean else "  "
                cell   = f"{marker}{m:.4f} +/- {ci:.4f}{marker}"
            row += f"  {cell:^{col_w}}"
        lines.append(row)
    lines.append("-" * 100)

    # ── Delta rows ─────────────────────────────────────────────────────────────
    if len(models) >= 3:
        orig_key, hub_key, clef_key = models[0], models[1], models[2]
        lines.append("")
        lines.append("IMPROVEMENT (positive = better)")
        lines.append("-" * 80)
        for label, a_key, b_key in [
            ("Delta_Huber  (Huber - original)", hub_key,  orig_key),
            ("Delta_CLEF   (CLEF  - Huber)",    clef_key, hub_key),
        ]:
            row = f"{label:<35}"
            for m in METRICS:
                sign  = 1 if LOWER_BETTER[m] else -1
                delta = sign * (summary[b_key][m]["mean"] - summary[a_key][m]["mean"])
                row  += f"  {METRIC_LABELS[m]}: {delta:+.4f}   "
            lines.append(row)
        lines.append("-" * 80)

    # ── Per-fold breakdown ─────────────────────────────────────────────────────
    lines.append("")
    lines.append("PER-FOLD BREAKDOWN")
    for mk in models:
        lines.append("")
        lines.append(f"  [{summary[mk]['label']}]")
        fold_metrics = all_results[mk]
        header2 = f"    {'Fold':<6}" + "".join(f"  {METRIC_LABELS[m]:<14}" for m in METRICS)
        lines.append(header2)
        for fm in fold_metrics:
            row2 = f"    {fm.get('fold', '?'):<6}"
            for m in METRICS:
                v = fm.get(m, float("nan"))
                row2 += f"  {v:>14.4f}"
            lines.append(row2)

    txt_path = os.path.join(output_dir, "results_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved results report -> {txt_path}")

    # ── Flat per-fold JSON ─────────────────────────────────────────────────────
    flat = {}
    for mk in models:
        flat[mk] = {
            "label":   summary[mk]["label"],
            "summary": {m: {"mean": summary[mk][m]["mean"],
                            "ci":   summary[mk][m]["ci_margin"]}
                        for m in METRICS},
            "per_fold": [{m: fm.get(m) for m in METRICS + ["fold", "subjects"]}
                         for fm in all_results[mk]],
        }
    json_path = os.path.join(output_dir, "per_fold_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(flat, f, indent=2, default=str)
    print(f"  Saved per-fold JSON  -> {json_path}")


def _print_comparison_table(summary: dict) -> None:
    """Print a formatted comparison table to stdout."""
    models = list(summary.keys())
    col_w  = 26

    header = f"{'Metric':<22}"
    for mk in models:
        header += f"  {summary[mk]['label']:^{col_w}}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for metric in METRICS:
        row = f"{METRIC_LABELS[metric]:<22}"
        best_mean = None
        for mk in models:
            m = summary[mk][metric]["mean"]
            if not np.isnan(m):
                if best_mean is None:
                    best_mean = m
                elif LOWER_BETTER[metric] and m < best_mean:
                    best_mean = m
                elif not LOWER_BETTER[metric] and m > best_mean:
                    best_mean = m

        for mk in models:
            m  = summary[mk][metric]["mean"]
            ci = summary[mk][metric]["ci_margin"]
            if np.isnan(m):
                cell = "N/A"
            else:
                cell = f"{m:.4f} ± {ci:.4f}"
                if m == best_mean:
                    cell = f"**{cell}**"
            row += f"  {cell:^{col_w}}"
        print(row)
    print("=" * len(header))


def _plot_side_by_side(summary: dict, output_dir: str) -> None:
    """One figure per metric: side-by-side bars with CI for all model variants."""
    fig_dir = os.path.join(output_dir, "figures")
    models  = list(summary.keys())
    labels  = [summary[mk]["label"] for mk in models]
    # original=grey, Huber=green, CLEF=orange
    colors  = ["#b0b0b0", "#a8d5a2", "#e8906a"]

    for metric in METRICS:
        means  = [summary[mk][metric]["mean"]      for mk in models]
        errors = [summary[mk][metric]["ci_margin"] for mk in models]

        fig, ax = plt.subplots(figsize=(5, 4))
        x       = np.arange(len(models))
        bars    = ax.bar(x, means, yerr=errors, capsize=7,
                         color=colors[:len(models)], edgecolor="black",
                         linewidth=0.7, width=0.45)

        # Annotate bars with mean values
        for bar, mean, err in zip(bars, means, errors):
            if not np.isnan(mean):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        mean + err + 0.001,
                        f"{mean:.4f}", ha="center", va="bottom", fontsize=8)

        # Highlight the better model
        lower_better = LOWER_BETTER[metric]
        valid = [(i, m) for i, m in enumerate(means) if not np.isnan(m)]
        if valid:
            best_i = min(valid, key=lambda t: t[1] if lower_better else -t[1])[0]
            bars[best_i].set_edgecolor("#2d7d46")
            bars[best_i].set_linewidth(2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
        ax.set_title(f"{METRIC_LABELS[metric]}", fontsize=10)
        ax.grid(True, axis="y", alpha=0.35)
        plt.tight_layout()

        save_path = os.path.join(fig_dir, f"cmp_{metric}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {save_path}")

    # Combined panel figure — 7 metrics in a 2×4 grid (last cell unused)
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    for ax, metric in zip(axes.flat, METRICS):
        means  = [summary[mk][metric]["mean"]      for mk in models]
        errors = [summary[mk][metric]["ci_margin"] for mk in models]
        x      = np.arange(len(models))
        ax.bar(x, means, yerr=errors, capsize=5, color=colors[:len(models)],
               edgecolor="black", linewidth=0.6, width=0.45)
        ax.set_xticks(x)
        short = ["RHN\n(orig)", "RHN\n+Huber", "RHN\n+CLEF"]
        ax.set_xticklabels(short[:len(models)], fontsize=8)
        ax.set_title(METRIC_LABELS[metric], fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    # Hide the spare 8th cell
    for ax in axes.flat[len(METRICS):]:
        ax.set_visible(False)

    fig.suptitle("ReHeartNet: MSE vs. +Huber vs. +Huber+CLEF  (8-fold CV, BIDMC)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    panel_path = os.path.join(fig_dir, "comparison_panel.png")
    plt.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved panel -> {panel_path}")


# ------------------------------------------------------------------------------
# Publication-quality outputs: LaTeX table + paper-grade figures
# ------------------------------------------------------------------------------

_LATEX_METRIC_HEADERS = {
    "rmse":            r"RMSE$^\dagger$ $\downarrow$",
    "prd":             r"PRD (\%) $\downarrow$",
    "pearson_r":       r"Pearson $r$ $\uparrow$",
    "bce":             r"BCE $\downarrow$",
    "emd":             r"EMD (s) $\downarrow$",
    "ks_stat":         r"KS stat $\downarrow$",
    "beat_timing_mae": r"Beat MAE (s) $\downarrow$",
}


def _save_latex_table(summary: dict, output_dir: str) -> None:
    """Write a ready-to-paste LaTeX table to figures/comparison_table.tex.

    The file can be included in main.tex via:
        \\input{results/comparison_reheartnet/figures/comparison_table.tex}
    """
    fig_dir = os.path.join(output_dir, "figures")
    models  = list(summary.keys())
    metrics = list(_LATEX_METRIC_HEADERS.keys())

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"    \centering")
    lines.append(r"    \caption{Three-way comparison on BIDMC (8-fold CV, mean $\pm$ 95\% CI). "
                 r"$\downarrow$~lower~better; $\uparrow$~higher~better. \textbf{Bold}~=~best.}")
    lines.append(r"    \label{tab:comparison_reheartnet_auto}")
    lines.append(r"    \resizebox{\textwidth}{!}{%")
    lines.append(r"    \begin{tabular}{l" + "c" * len(models) + "}")
    lines.append(r"        \toprule")

    # Header row
    hdr = "        Model"
    for mk in models:
        hdr += f" & {summary[mk]['label']}"
    lines.append(hdr + r" \\")
    lines.append(r"        \midrule")

    # Data rows
    for m in metrics:
        vals = [(mk, summary[mk][m]["mean"], summary[mk][m]["ci_margin"]) for mk in models]
        low_better = LOWER_BETTER[m]
        best_mk = min(vals, key=lambda x: x[1] if low_better else -x[1],
                      default=(None, 0, 0))[0]
        row = f"        {_LATEX_METRIC_HEADERS[m]}"
        for mk, mean, ci in vals:
            cell = f"{mean:.4f} $\\pm$ {ci:.4f}"
            if mk == best_mk:
                cell = f"\\textbf{{{mean:.4f}}} $\\pm$ {ci:.4f}"
            row += f" & {cell}"
        lines.append(row + r" \\")

    # Delta rows
    lines.append(r"        \midrule")
    orig_key, hub_key, clef_key = models[0], models[1], models[2]
    for label, a_key, b_key in [
        (r"$\Delta_\text{Huber}$", hub_key, orig_key),
        (r"$\Delta_\text{CLEF}$",  clef_key, hub_key),
    ]:
        row = f"        {label}"
        for m in metrics:
            sign  = 1 if LOWER_BETTER[m] else -1
            delta = sign * (summary[b_key][m]["mean"] - summary[a_key][m]["mean"])
            color = "green!60!black" if delta >= 0 else "red!60!black"
            row += f" & {{\\color{{{color}}}{delta:+.4f}}}"
        lines.append(row + r" \\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}}")
    lines.append(r"    \\\small $^\dagger$RMSE on z-scored signals; not directly comparable to "
                 r"Lee et al.\ (mV).")
    lines.append(r"\end{table}")

    path = os.path.join(fig_dir, "comparison_table.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved LaTeX table -> {path}")


def _plot_paper_reconstruction(all_samples: dict, output_dir: str,
                               fs: int = 125, n_windows: int = 4) -> None:
    """Publication-quality ECG reconstruction figure.

    Layout: one column per example window.  Each subplot overlays GT (blue) with
    each model's prediction (grey / green / orange) so differences are immediately
    visible.  Time axis in seconds.  Compact legend in the first subplot only.
    """
    fig_dir = os.path.join(output_dir, "figures")
    models  = list(all_samples.keys())
    if not models:
        return

    n_ex    = min(n_windows, min(len(s["pred"]) for s in all_samples.values()))
    if n_ex == 0:
        return

    pred_colors  = {"reheartnet_original": "#888888",
                    "reheartnet_huber":    "#2e9e4a",
                    "reheartnet_clef":     "#d45f0e"}
    pred_labels  = {mk: MODELS[mk]["label"].replace("ReHeartNet", "RHN")
                    for mk in models}
    gt_color     = "#1f77b4"

    fig, axes = plt.subplots(1, n_ex, figsize=(4.5 * n_ex, 3.2), squeeze=False)

    for col in range(n_ex):
        ax = axes[0][col]

        # Ground truth — use CLEF model's true if available (longer window), else first model
        ref_key     = "reheartnet_clef" if "reheartnet_clef" in all_samples else models[0]
        true_sig    = all_samples[ref_key]["true"][col]
        w_sec       = all_samples[ref_key]["window_sec"] or 10.0
        t_ref       = np.linspace(0, w_sec, len(true_sig))

        ax.plot(t_ref, true_sig, color=gt_color, linewidth=1.3, label="Ground truth",
                zorder=3, alpha=0.9)

        # Each model's prediction
        for mk in models:
            pred_sig = all_samples[mk]["pred"][col]
            w        = all_samples[mk]["window_sec"] or 10.0
            t        = np.linspace(0, w, len(pred_sig))
            color    = pred_colors.get(mk, "#999999")
            ax.plot(t, pred_sig, color=color, linewidth=0.9, linestyle="--",
                    label=pred_labels[mk], zorder=2, alpha=0.85)

        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_xlim(0, max(w_sec, max(
            all_samples[mk]["window_sec"] or 10.0 for mk in models)))
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        ax.set_title(f"Window {col + 1}", fontsize=9)
        if col == 0:
            ax.set_ylabel("ECG (z-scored)", fontsize=8)
            ax.legend(fontsize=6.5, loc="upper right", framealpha=0.7)

    fig.suptitle("ECG reconstruction: ground truth vs. model predictions  (fold 0)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(fig_dir, "paper_reconstruction.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved paper reconstruction -> {path}")


# ------------------------------------------------------------------------------
# Additional comparison figures for the paper
# ------------------------------------------------------------------------------

def _plot_delta_bars(summary: dict, output_dir: str) -> None:
    """Signed improvement bars: Delta_Huber and Delta_CLEF per metric.

    Positive = improvement (lower for ↓ metrics, higher for ↑).
    Grey bars = Huber - original, orange bars = CLEF - Huber.
    """
    fig_dir = os.path.join(output_dir, "figures")
    keys    = list(summary.keys())          # [original, huber, clef]
    if len(keys) < 3:
        return
    orig_key, hub_key, clef_key = keys[0], keys[1], keys[2]

    metrics  = [m for m in METRICS if m != "ks_stat"]   # KS signed change is less intuitive
    n        = len(metrics)
    x        = np.arange(n)
    width    = 0.35

    delta_huber = []
    delta_clef  = []
    for m in metrics:
        o = summary[orig_key][m]["mean"]
        h = summary[hub_key][m]["mean"]
        c = summary[clef_key][m]["mean"]
        sign = 1 if LOWER_BETTER[m] else -1   # positive = improvement
        delta_huber.append(sign * (h - o))
        delta_clef.append(sign * (c - h))

    fig, ax = plt.subplots(figsize=(11, 4))
    bars_h = ax.bar(x - width / 2, delta_huber, width, label="$\\Delta$ Huber (Huber - original)",
                    color="#a8d5a2", edgecolor="black", linewidth=0.7)
    bars_c = ax.bar(x + width / 2, delta_clef,  width, label="$\\Delta$ CLEF (CLEF - Huber)",
                    color="#e8906a", edgecolor="black", linewidth=0.7)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], fontsize=9)
    ax.set_ylabel("Signed improvement (positive = better)", fontsize=9)
    ax.set_title("Per-metric improvement over baseline  (positive = better)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, "delta_improvement.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_per_fold_boxes(all_results: dict, output_dir: str) -> None:
    """Box plots: per-fold metric distribution, one box per model."""
    fig_dir = os.path.join(output_dir, "figures")
    models  = list(all_results.keys())
    colors  = ["#b0b0b0", "#a8d5a2", "#e8906a"]
    labels  = [MODELS[mk]["label"] for mk in models]
    n_cols  = 4
    n_rows  = (len(METRICS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))

    for ax, metric in zip(axes.flat, METRICS):
        data = [[m.get(metric, float("nan")) for m in all_results[mk]] for mk in models]
        data = [[v for v in d if not (isinstance(v, float) and np.isnan(v))] for d in data]
        bp   = ax.boxplot(data, patch_artist=True, widths=0.5,
                          medianprops={"color": "black", "linewidth": 1.5})
        for patch, color in zip(bp["boxes"], colors[:len(models)]):
            patch.set_facecolor(color)
        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels(["orig", "+Huber", "+CLEF"][:len(models)], fontsize=8)
        ax.set_title(METRIC_LABELS[metric], fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    for ax in axes.flat[len(METRICS):]:
        ax.set_visible(False)

    fig.suptitle("Per-fold metric distributions  (8-fold CV, BIDMC)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(fig_dir, "per_fold_boxes.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_waveform_comparison(all_samples: dict, output_dir: str, fs: int = 125) -> None:
    """Multi-model waveform overlay: rows = models, columns = example windows.

    Uses time in seconds on the x-axis so models with different window lengths
    can be displayed side-by-side.  Up to 4 example windows are shown.
    """
    fig_dir = os.path.join(output_dir, "figures")
    models  = list(all_samples.keys())
    n_ex    = min(4, min(len(s["pred"]) for s in all_samples.values()))
    if n_ex == 0:
        return

    n_rows = len(models) + 1   # ground truth + one row per model
    fig, axes = plt.subplots(n_rows, n_ex, figsize=(4 * n_ex, 2 * n_rows),
                             squeeze=False)

    for col in range(n_ex):
        # Ground truth row — use first model's true signal
        first_key = models[0]
        true_sig  = all_samples[first_key]["true"][col]
        win_sec   = all_samples[first_key]["window_sec"] or 10.0
        t = np.linspace(0, win_sec, len(true_sig))
        axes[0][col].plot(t, true_sig, color="steelblue", linewidth=0.8)
        if col == 0:
            axes[0][col].set_ylabel("Ground\ntruth", fontsize=8, rotation=0, labelpad=40)
        axes[0][col].set_title(f"Window {col + 1}", fontsize=9)
        axes[0][col].set_xlim(0, win_sec)
        axes[0][col].tick_params(labelsize=7)

        # One row per model
        row_colors = ["#b0b0b0", "#a8d5a2", "#e8906a"]
        for row, (mk, color) in enumerate(zip(models, row_colors), start=1):
            pred_sig = all_samples[mk]["pred"][col]
            true_sig = all_samples[mk]["true"][col]
            w        = all_samples[mk]["window_sec"] or 10.0
            t_model  = np.linspace(0, w, len(pred_sig))
            axes[row][col].plot(t_model, true_sig,  color="steelblue", linewidth=0.6,
                                alpha=0.5, label="GT")
            axes[row][col].plot(t_model, pred_sig,  color=color,       linewidth=0.8,
                                label="pred")
            if col == 0:
                short = MODELS[mk]["label"].replace("ReHeartNet", "RHN")
                axes[row][col].set_ylabel(short, fontsize=7, rotation=0, labelpad=50)
            axes[row][col].set_xlim(0, w)
            axes[row][col].tick_params(labelsize=7)
            if row == n_rows - 1:
                axes[row][col].set_xlabel("Time (s)", fontsize=8)

    fig.suptitle("ECG reconstruction quality comparison  (fold 0 test windows)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(fig_dir, "waveform_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_rr_kde_comparison(all_samples: dict, output_dir: str, fs: int = 125) -> None:
    """KDE of RR intervals for ground truth and all models on the same axis."""
    try:
        import neurokit2 as nk       # noqa: PLC0415
    except ImportError:
        return

    fig_dir = os.path.join(output_dir, "figures")
    models  = list(all_samples.keys())
    colors  = {"gt": "steelblue",
               models[0]: "#b0b0b0",
               models[1]: "#4caf50",
               models[2]: "#e8906a"}

    def _rr(windows: np.ndarray) -> np.ndarray:
        rr_all = []
        for w in windows:
            try:
                _, info = nk.ecg_peaks(w.astype(float), sampling_rate=fs,
                                       method="pantompkins1985")
                peaks = info["ECG_R_Peaks"]
                if len(peaks) >= 3:
                    rr_all.extend(np.diff(peaks) / fs)
            except Exception:
                pass
        return np.array(rr_all)

    import warnings
    fig, ax = plt.subplots(figsize=(7, 4))
    # Ground truth (from first model's true signal)
    first_key = models[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rr_gt = _rr(all_samples[first_key]["true"])
    if len(rr_gt) >= 5:
        from scipy.stats import gaussian_kde   # noqa: PLC0415
        kde = gaussian_kde(rr_gt, bw_method=0.2)
        xs  = np.linspace(0.3, 1.8, 300)
        ax.plot(xs, kde(xs), color="steelblue", linewidth=2, label="Ground truth")
        ax.fill_between(xs, kde(xs), alpha=0.15, color="steelblue")

    # Each model's predictions
    model_colors = ["#888888", "#4caf50", "#e8906a"]
    for mk, col in zip(models, model_colors):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rr_pred = _rr(all_samples[mk]["pred"])
        if len(rr_pred) >= 5:
            from scipy.stats import gaussian_kde   # noqa: PLC0415
            kde = gaussian_kde(rr_pred, bw_method=0.2)
            ax.plot(xs, kde(xs), color=col, linewidth=1.5,
                    linestyle="--", label=MODELS[mk]["label"].replace("ReHeartNet", "RHN"))

    ax.set_xlabel("RR interval (s)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("RR interval distributions  (fold 0 test windows)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, "rr_kde_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ------------------------------------------------------------------------------
# Reconstruction overlay: one figure showing both models on the same window
# ------------------------------------------------------------------------------

def _save_recon_figure(model, test_loader, test_subs, device,
                       fold_idx, fig_dir) -> None:
    model.eval()
    with torch.no_grad():
        ppg_b, ecg_b = next(iter(test_loader))
        pred_b = model(ppg_b.to(device)).cpu()

    true_np = ecg_b.squeeze(-1).numpy()
    pred_np = pred_b.squeeze(-1).numpy()

    plot_reconstruction_samples(
        true_np, pred_np,
        subject_ids=[test_subs[min(i, len(test_subs)-1)] for i in range(min(4, len(true_np)))],
        fold_idx=fold_idx,
        save_path=os.path.join(fig_dir, f"recon_fold{fold_idx:02d}.png"),
    )


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    # Per-variant subdirectories (checkpoints/partial/summary/figures) are created
    # lazily inside _run_model, keyed by model_key — see run_dir below.

    device = config.DEVICE
    print(f"Device: {device}")

    # Auto-select CLEF size: medium on GPU (better features), small on CPU
    if args.clef_size == "auto":
        args.clef_size = "medium" if device.type == "cuda" else "small"
        print(f"CLEF size auto-selected: {args.clef_size}")
    # Auto-construct path if not given
    if args.clef_path is None:
        args.clef_path = os.path.join(args.clef_dir, f"clef_{args.clef_size}.ckpt")
        print(f"CLEF path auto-set: {args.clef_path}")

    # Load best Optuna params (used as defaults for our three variants)
    best_params_path = os.path.join("results", "best_hyperparams.json")
    best_params: dict = {}
    if os.path.exists(best_params_path):
        with open(best_params_path) as f:
            best_params = json.load(f)
        print(f"Loaded hyperparameters from {best_params_path}")
    else:
        print("No best_hyperparams.json -- using config defaults.")

    # Build shared frozen encoders (once for the whole run)
    print(f"Loading CLEF encoder ({args.clef_size}) ...")
    clef_encoder = load_clef_encoder(args.clef_path, args.clef_size, device)
    ptbxl_clf    = get_ptbxl_classifier(args.classifier_path).to(device)

    all_subjects = get_all_record_names()

    # All three variants share the same subject-grouped 8-fold CV splits
    # (subjects are fully separated between folds -- no subject leakage).
    n_folds = 2 if args.dry_run else args.n_folds
    splits = get_cv_splits(
        all_subjects, n_splits=n_folds,
        save_path=os.path.join(args.output_dir, "fold_assignments.json"),
    )

    # -- Run all three model variants -----------------------------------------
    selected = MODELS
    if args.only:
        keys = [k.strip() for k in args.only.split(",")]
        unknown = [k for k in keys if k not in MODELS]
        if unknown:
            raise ValueError(f"Unknown model key(s) in --only: {unknown}. Valid keys: {list(MODELS)}")
        selected = {k: MODELS[k] for k in keys}

    all_results = {}
    all_samples  = {}
    for model_key, model_cfg in selected.items():
        print(f"\n{'-'*60}")
        print(f"  Model : {model_cfg['label']}")
        print(f"{'-'*60}")
        fold_metrics, sample = _run_model(
            model_key, model_cfg, args, device,
            clef_encoder, ptbxl_clf, splits, best_params,
        )
        all_results[model_key] = fold_metrics
        all_samples[model_key]  = sample

    if args.only:
        print(f"\n--only was given ({list(selected)}); skipping comparison-output assembly. "
              f"Rerun with --resume and no --only once all variants have completed to assemble the report.")
        return

    # -- Comparison output ----------------------------------------------------
    print("\n" + "-"*60)
    print("  Building comparison outputs")
    print("-"*60)
    summary = _save_comparison_json(all_results, args.output_dir)
    _print_comparison_table(summary)
    _save_results_report(summary, all_results, args.output_dir)
    _log_wandb_comparison(summary, all_results, args)
    _plot_side_by_side(summary, args.output_dir)
    _plot_delta_bars(summary, args.output_dir)
    _plot_per_fold_boxes(all_results, args.output_dir)
    _save_latex_table(summary, args.output_dir)
    _plot_paper_reconstruction(all_samples, args.output_dir)
    _plot_waveform_comparison(all_samples, args.output_dir)
    _plot_rr_kde_comparison(all_samples, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")
    print("Key files:")
    print(f"  {args.output_dir}/results_report.txt                      <- human-readable table + per-fold breakdown")
    print(f"  {args.output_dir}/per_fold_results.json                  <- flat per-fold JSON")
    print(f"  {args.output_dir}/comparison.json                        <- full nested JSON")
    print(f"  {args.output_dir}/figures/comparison_table.tex          <- paste into LaTeX paper")
    print(f"  {args.output_dir}/figures/paper_reconstruction.png      <- GT vs models overlay (paper figure)")
    print(f"  {args.output_dir}/figures/comparison_panel.png          <- 2x4 panel (all 7 metrics)")
    print(f"  {args.output_dir}/figures/delta_improvement.png         <- signed Huber/CLEF gains")
    print(f"  {args.output_dir}/figures/per_fold_boxes.png            <- box plots per metric/model")
    print(f"  {args.output_dir}/figures/rr_kde_comparison.png         <- RR interval KDE overlay")
    print(f"\nAll models evaluated with {n_folds}-fold subject-grouped CV ({len(all_subjects)} subjects).")


if __name__ == "__main__":
    main()
