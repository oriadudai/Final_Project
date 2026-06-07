"""Quick sanity check -- runs in ~5-8 minutes on CPU.

Uses 3 BIDMC subjects, 2 manual folds, 3 epochs.
Tests the full pipeline end-to-end:
  preprocessing -> data loading -> training (Huber+CLEF) -> evaluation -> plots

Usage:
    python scripts/sanity_check.py
    python scripts/sanity_check.py --clef-path models/clef/clef_small.ckpt
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import tempfile
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.config as config

CLEF_PATH   = "models/clef/clef_small.ckpt"
SUBJECTS    = ["bidmc01", "bidmc02", "bidmc03"]
EPOCHS      = 3
BATCH_SIZE  = 16   # faster than paper's 1; batch_size=1 tested explicitly in section 2

METRIC_KEYS = ("rmse", "prd", "pearson_r", "bce", "emd", "ks_stat", "ks_pvalue", "beat_timing_mae")


def separator(title=""):
    w = 55
    if title:
        pad = (w - len(title) - 2) // 2
        print(f"\n{'-'*pad} {title} {'-'*pad}")
    else:
        print("-" * w)


def _is_bad(v) -> bool:
    return v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


def record(problems: list, msg: str) -> None:
    problems.append(msg)
    print(f"     !! PROBLEM: {msg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clef-path", default=CLEF_PATH)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("cpu")
    print(f"Device : {device}")
    print(f"Subjects: {SUBJECTS}")
    print(f"Epochs  : {EPOCHS}  |  Batch size: {BATCH_SIZE}")

    problems = []

    # -- 1. Imports -----------------------------------------------------------
    separator("1 - Imports")
    from core.data_loader import (
        build_group_fold, build_test_dataset, get_cv_splits, split_train_val,
    )
    from core.evaluate import evaluate_fold
    from core.losses.composite_loss import ClinicalCompositeLoss, load_clef_encoder
    from core.metrics.clinical_metrics import compute_bce, extract_rr_intervals
    from core.models.baselines import get_model
    from core.models.ptbxl_classifier import build_classifier, get_ptbxl_classifier
    from core.train import train_fold
    from core.visualization.plots import (
        plot_fold_metrics_bar,
        plot_loss_curves,
        plot_metric_boxplots,
        plot_reconstruction_samples,
        plot_rr_distributions,
        save_results_summary,
    )
    from src.preprocessing import build_subject_windows, get_all_record_names
    import optuna
    import wandb
    print(f"  optuna {optuna.__version__} OK")
    print(f"  wandb {wandb.__version__} OK")
    print("  All imports OK")

    # -- 2. CLEF encoder + LR schedule + batch_size=1 smoke test -------------
    separator("2 - CLEF encoder")
    if not os.path.exists(args.clef_path):
        print(f"  ERROR: CLEF checkpoint not found at '{args.clef_path}'", file=sys.stderr)
        print("  Run: python scripts/download_clef.py  (or pass --clef-path)", file=sys.stderr)
        sys.exit(1)

    clef_encoder = load_clef_encoder(args.clef_path, model_size="small", device=device)
    dummy = torch.randn(2, 1, 5000)
    feat  = clef_encoder(dummy)
    print(f"  Loaded  |  input (2,1,5000) -> features {tuple(feat.shape)}")

    ptbxl_clf = get_ptbxl_classifier(pretrained_path=None).to(device)
    print("  PTB-XL surrogate classifier ready")

    # Verify LR schedule fires correctly at epoch 50
    _m     = torch.nn.Linear(1, 1)
    _opt   = optim.Adam(_m.parameters(), lr=1e-3)
    _sched = optim.lr_scheduler.LambdaLR(_opt, lr_lambda=lambda e: 0.75 ** (e // 50))
    _opt.step()   # must come before sched.step() per PyTorch >=1.1 API
    for _ in range(51):
        _sched.step()
    _lr       = _opt.param_groups[0]["lr"]
    _expected = 1e-3 * 0.75
    if abs(_lr - _expected) > 1e-9:
        record(problems, f"LR schedule: expected {_expected:.6f} after 51 steps, got {_lr:.6f}")
    else:
        print(f"  LR schedule OK: fires at epoch 50  (1e-3 -> {_lr:.2e})")

    # Verify ClinicalCompositeLoss with batch_size=1 (paper's actual batch size)
    _model_1 = get_model("reheartnet", hidden_size=config.HIDDEN_SIZE).to(device)
    _crit_1  = ClinicalCompositeLoss(clef_encoder, lambda_clinical=0.1, huber_delta=1.0)
    _ppg_1   = torch.randn(1, config.SEQ_LEN, 1)
    _ecg_1   = torch.randn(1, config.SEQ_LEN, 1)
    _pred_1  = _model_1(_ppg_1)
    _loss_1  = _crit_1(_pred_1, _ecg_1)
    if _is_bad(_loss_1.item()):
        record(problems, f"ClinicalCompositeLoss batch_size=1: loss={_loss_1}")
    else:
        _loss_1.backward()
        print(f"  ClinicalCompositeLoss batch_size=1 forward+backward OK  loss={_loss_1.item():.4f}")
    del _model_1, _crit_1, _ppg_1, _ecg_1, _pred_1, _loss_1

    # -- 3. Preprocessing + subject / split enumeration -----------------------
    separator("3 - Preprocessing (3 subjects)")

    # Standard 10s windows (our protocol)
    for subj in SUBJECTS:
        ppg, ecg = build_subject_windows(subj, apply_phase_align=False)
        print(f"  {subj}: ppg={ppg.shape}  ecg={ecg.shape}  "
              f"min/max ecg=[{ecg.min():.2f}, {ecg.max():.2f}]")
        if ppg.shape[0] == 0 or ecg.shape[0] == 0:
            record(problems, f"{subj}: no windows returned from preprocessing")

    # Lee et al. protocol: 4 s windows, no overlap, bandpass filter
    ppg_4s, ecg_4s = build_subject_windows(
        "bidmc01", apply_phase_align=False,
        window_sec=4.0, overlap_frac=0.0, apply_bandpass=True,
    )
    print(f"  bidmc01 (4s/overlap=0/bandpass): ppg={ppg_4s.shape}  ecg={ecg_4s.shape}  "
          f"min/max ecg=[{ecg_4s.min():.2f}, {ecg_4s.max():.2f}]")
    expected_win = 500   # 4 s * 125 Hz
    if ppg_4s.shape[1] != expected_win:
        record(problems, f"4s windows: expected window_size={expected_win}, got {ppg_4s.shape[1]}")
    if ppg_4s.shape[0] == 0:
        record(problems, "4s windows: no windows produced with overlap_frac=0.0")

    # build_test_dataset
    test_ds_10s = build_test_dataset(["bidmc01"])
    print(f"  build_test_dataset: {len(test_ds_10s)} windows  "
          f"ppg shape={test_ds_10s[0][0].shape}")
    if len(test_ds_10s) == 0:
        record(problems, "build_test_dataset: returned 0 windows")

    all_subs = get_all_record_names()
    print(f"  get_all_record_names: {len(all_subs)} subjects")
    if len(all_subs) != 53:
        record(problems, f"get_all_record_names: expected 53, got {len(all_subs)}")

    splits = get_cv_splits(
        all_subs, n_splits=8,
        save_path="results/sanity_check/fold_assignments.json",
    )
    print(f"  get_cv_splits: {len(splits)} folds  "
          f"test sizes = {[len(s[1]) for s in splits[:3]]}...")
    if len(splits) != 8:
        record(problems, f"get_cv_splits: expected 8 folds, got {len(splits)}")

    # best_hyperparams.json load + type-cast (mirrors run_cv._load_best_hyperparams)
    _dummy_hp = {"lr": 5e-4, "lambda_clinical": 0.05, "huber_delta": 0.8,
                 "hidden_size": 64, "epochs": 800}
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir="results/sanity_check"
    ) as _f:
        json.dump(_dummy_hp, _f)
        _tmp_hp_path = _f.name
    try:
        with open(_tmp_hp_path) as _f:
            _loaded = json.load(_f)
        assert float(_loaded["lr"]) == 5e-4
        assert int(_loaded["hidden_size"]) == 64
        print("  best_hyperparams JSON load + type-cast OK")
    finally:
        os.unlink(_tmp_hp_path)

    # -- 4. 2-fold mini CV ----------------------------------------------------
    separator("4 - 2-fold mini CV")
    folds = [
        (["bidmc02", "bidmc03"], ["bidmc01"]),
        (["bidmc01", "bidmc03"], ["bidmc02"]),
    ]

    fold_metrics = []
    os.makedirs("results/sanity_check/figures", exist_ok=True)

    for fold_idx, (train_subs, test_subs) in enumerate(folds):
        print(f"\n  -- Fold {fold_idx} | train={train_subs} | test={test_subs}")

        train_ds, test_ds = build_group_fold(train_subs, test_subs, apply_align=True)
        train_inner, val_ds = split_train_val(train_ds, val_fraction=0.1)

        print(f"     windows -> train:{len(train_inner)}  val:{len(val_ds)}  test:{len(test_ds)}")

        if len(train_inner) == 0:
            record(problems, f"fold {fold_idx}: training set is empty")
        if len(val_ds) == 0:
            record(problems, f"fold {fold_idx}: validation set is empty")
        if len(test_ds) == 0:
            record(problems, f"fold {fold_idx}: test set is empty")

        train_loader = DataLoader(train_inner, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,      batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,     batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

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
            hidden_size     = 32,
            checkpoint_dir  = "results/sanity_check/checkpoints",
            use_wandb       = False,
            early_stop_patience = None,
            loss_type       = "clef",
        )

        train_strs, val_strs = [], []
        for i, v in enumerate(history["train"]):
            if _is_bad(v):
                record(problems, f"fold {fold_idx} train loss[{i}] = {v}")
                train_strs.append(str(v))
            else:
                train_strs.append(f"{v:.4f}")
        for i, v in enumerate(history["val"]):
            if _is_bad(v):
                record(problems, f"fold {fold_idx} val loss[{i}] = {v}")
                val_strs.append(str(v))
            else:
                val_strs.append(f"{v:.4f}")

        print(f"     train loss history: {train_strs}")
        print(f"     val   loss history: {val_strs}")

        plot_loss_curves(
            history["train"], history["val"], fold_idx,
            save_path=f"results/sanity_check/figures/loss_fold{fold_idx}.png",
        )

        print("     Evaluating ...")
        metrics = evaluate_fold(model, test_loader, ptbxl_clf, device, clef_encoder=clef_encoder)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

        for key in METRIC_KEYS:
            val = metrics.get(key)
            if _is_bad(val):
                record(problems, f"fold {fold_idx} {key} = {val}")
                print(f"     {key:<20} = {val}")
            else:
                print(f"     {key:<20} = {val:.4f}")

        # Collect all test predictions for visualisation
        model.eval()
        all_true_np, all_pred_np = [], []
        with torch.no_grad():
            for ppg_t, ecg_t in test_loader:
                pred_t = model(ppg_t.to(device)).cpu()
                all_true_np.append(ecg_t.squeeze(-1).numpy())
                all_pred_np.append(pred_t.squeeze(-1).numpy())
        all_true_np = np.concatenate(all_true_np, axis=0)
        all_pred_np = np.concatenate(all_pred_np, axis=0)

        plot_reconstruction_samples(
            all_true_np, all_pred_np,
            fold_idx=fold_idx,
            n_samples=2,
            save_path=f"results/sanity_check/figures/recon_fold{fold_idx}.png",
        )

        rr_true = extract_rr_intervals(all_true_np)
        rr_pred = extract_rr_intervals(all_pred_np)
        if len(rr_true) < 2:
            record(problems, f"fold {fold_idx} rr_true: {len(rr_true)} intervals detected (too few)")
        if len(rr_pred) < 2:
            record(problems, f"fold {fold_idx} rr_pred: {len(rr_pred)} intervals detected (too few)")
        plot_rr_distributions(
            rr_true, rr_pred,
            subject_id=f"fold{fold_idx}",
            fold_idx=fold_idx,
            save_path=f"results/sanity_check/figures/rr_dist_fold{fold_idx}.png",
        )

    # Aggregate plots
    print("\n  Aggregate plots ...")
    for key in METRIC_KEYS:
        if key == "ks_pvalue":
            continue
        plot_fold_metrics_bar(
            fold_metrics, key,
            save_path=f"results/sanity_check/figures/metric_bar_{key}.png",
        )
    plot_metric_boxplots(
        fold_metrics,
        save_path="results/sanity_check/figures/metric_boxplots.png",
    )
    print("  Aggregate plots OK")

    # Verify fold_metrics is JSON-serializable (run_cv saves this after every fold)
    json.dumps(fold_metrics, default=str)
    print("  fold_metrics JSON-serializable OK")

    # -- 5. Baseline models (at config.HIDDEN_SIZE = 64, the real-run default) -
    separator("5 - Baseline models")
    dummy_ppg      = torch.randn(2, config.SEQ_LEN, 1)
    expected_shape = (2, config.SEQ_LEN, 1)
    for name in ("lstm", "bilstm", "reheartnet"):
        m   = get_model(name, hidden_size=config.HIDDEN_SIZE).to(device)
        out = m(dummy_ppg)
        if tuple(out.shape) != expected_shape:
            record(problems, f"{name}: output shape {tuple(out.shape)}, expected {expected_shape}")
        else:
            print(f"  {name:<12} forward pass OK  H={config.HIDDEN_SIZE}  shape={tuple(out.shape)}")

    # -- 5b. compare_reheartnet.py smoke test ---------------------------------
    separator("5b - compare_reheartnet")
    _cr_path = os.path.join(os.path.dirname(__file__), "compare_reheartnet.py")
    _spec = importlib.util.spec_from_file_location("compare_reheartnet", _cr_path)
    _cr   = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_cr)
    print("  compare_reheartnet import OK")

    _cr_dir = "results/sanity_check/compare_reheartnet"
    os.makedirs(os.path.join(_cr_dir, "figures"), exist_ok=True)

    # Minimal dummy inputs (3 model variants, 2 folds each)
    _cr_metrics = _cr.METRICS
    _dummy_results = {
        mk: [{"fold": fi, **{m: float(i * 0.1 + 0.1) for i, m in enumerate(_cr_metrics)}}
             for fi in range(2)]
        for mk in _cr.MODELS
    }
    _dummy_samples = {
        "reheartnet_original": {
            "true": np.random.randn(4, 500).astype(np.float32),
            "pred": np.random.randn(4, 500).astype(np.float32),
            "window_sec": 4.0,
        },
        "reheartnet_huber": {
            "true": np.random.randn(4, 500).astype(np.float32),
            "pred": np.random.randn(4, 500).astype(np.float32),
            "window_sec": 4.0,
        },
        "reheartnet_clef": {
            "true": np.random.randn(4, 1250).astype(np.float32),
            "pred": np.random.randn(4, 1250).astype(np.float32),
            "window_sec": None,
        },
    }

    # Build summary the same way compare_reheartnet.main() does:
    # _save_comparison_json both persists the JSON and returns the summary dict.
    _cr_summary = _cr._save_comparison_json(_dummy_results, _cr_dir)
    print("  _save_comparison_json OK")

    for fn_name, fn_args in [
        ("_print_comparison_table",    (_cr_summary,)),
        ("_save_results_report",       (_cr_summary, _dummy_results, _cr_dir)),
        ("_save_latex_table",          (_cr_summary, _cr_dir)),
        ("_plot_side_by_side",         (_cr_summary, _cr_dir)),
        ("_plot_delta_bars",           (_cr_summary, _cr_dir)),
        ("_plot_per_fold_boxes",       (_dummy_results, _cr_dir)),
        ("_plot_paper_reconstruction",  (_dummy_samples, _cr_dir)),
        ("_plot_waveform_comparison",   (_dummy_samples, _cr_dir)),
        ("_plot_rr_kde_comparison",     (_dummy_samples, _cr_dir)),
    ]:
        getattr(_cr, fn_name)(*fn_args)
        print(f"  {fn_name} OK")

    # Direct BCE path: build_classifier + compute_bce (compare_reheartnet calls these
    # outside evaluate_fold to compute BCE on 10 s windows for 4 s trained models)
    _clf_direct = build_classifier(clef_encoder).to(device)
    _true_10s   = np.random.randn(4, config.SEQ_LEN).astype(np.float32)
    _pred_10s   = np.random.randn(4, config.SEQ_LEN).astype(np.float32)
    _bce_direct = compute_bce(_true_10s, _pred_10s, _clf_direct, device)
    if _is_bad(_bce_direct):
        record(problems, f"direct compute_bce = {_bce_direct}")
    else:
        print(f"  direct compute_bce + build_classifier OK  bce={_bce_direct:.4f}")

    # -- 6. Summary -----------------------------------------------------------
    separator("6 - Summary")
    save_results_summary(
        fold_metrics,
        save_path="results/sanity_check/summary.json",
    )

    elapsed = time.time() - t0
    separator()

    if problems:
        print(f"\nSanity check FAILED -- {len(problems)} problem(s) found:")
        for p in problems:
            print(f"  !! {p}")
        sys.exit(1)

    print(f"Sanity check PASSED in {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"Figures -> results/sanity_check/figures/")
    print(f"Summary -> results/sanity_check/summary.json")


if __name__ == "__main__":
    main()
