"""Standalone post-hoc diagnostic-consistency evaluation (Guan-et-al.-style).

Adds a "does the reconstruction preserve enough diagnosis-relevant information
for a real pathology classifier to reach the same conclusion?" metric ON TOP OF
(never instead of) the existing CLEF-feature-space BCE metric in core/evaluate.py.

Why a separate script: run_cv.py / compare_reheartnet.py / core/evaluate.py are
the live training+eval pipeline (actively running CV jobs as of writing). This
script touches none of them — it only *reads* already-saved checkpoints
(checkpoints/<variant>/<arch>_fold_NN_best.pt) and the BIDMC test windows, and
writes its own results file. Run it once all the in-flight CV jobs have finished
(it would otherwise compete for GPU/IO with them, and there is nothing to gain
by running it earlier — every checkpoint it needs is written incrementally as
each fold's *_best.pt is saved).

Prerequisites (one-time; can be done in parallel with the current training runs
since they use a different dataset and write to a different checkpoint path):
    python scripts/download_ptbxl.py
    python scripts/train_diagnostic_classifier.py
        -> checkpoints/ptbxl_diagnostic_classifier.pt

Usage (after the reconstruction CV runs have completed):
    python scripts/evaluate_diagnostic_consistency.py \\
        --diagnostic-classifier checkpoints/ptbxl_diagnostic_classifier.pt \\
        --out results/diag_consistency.json

Merge the resulting diag_kl / diag_flip_rate values into your existing
rmse/prd/bce/emd/... summary tables -- no architecture changes, no relaunching.
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.config as config
from core.data_loader import build_test_dataset
from core.metrics.clinical_metrics import compute_confidence_interval
from core.losses.composite_loss import load_clef_encoder
from core.models.baselines import get_model
from core.models.ptbxl_classifier import CLEFProbeClassifier, SurrogateECGClassifier

_CKPT_RE = re.compile(r"(?P<arch>[a-zA-Z0-9]+)_fold_(?P<fold>\d+)_best\.pt$")
_DEFAULT_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


# ---------------------------------------------------------------------------
# Diagnostic classifier (trained by scripts/train_diagnostic_classifier.py)
# ---------------------------------------------------------------------------

def _load_diagnostic_classifier(path: str, clef_encoder=None):
    ckpt = torch.load(path, map_location="cpu")
    superclasses = ckpt.get("superclasses", _DEFAULT_SUPERCLASSES)
    clf_type = ckpt.get("type", "surrogate")
    if clf_type == "clef_probe":
        if clef_encoder is None:
            raise ValueError(f"Checkpoint {path} is type clef_probe — pass --clef-path.")
        model = CLEFProbeClassifier(clef_encoder, num_classes=len(superclasses))
        model.probe.load_state_dict(ckpt["probe_state_dict"])
        print(f"  Classifier: CLEFProbeClassifier (clef_size={ckpt.get('clef_size', '?')})")
    else:
        model = SurrogateECGClassifier(num_classes=len(superclasses))
        model.load_state_dict(ckpt["state_dict"])
        print(f"  Classifier: SurrogateECGClassifier")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, superclasses


def _diag_consistency(
    true_ecg: np.ndarray,
    pred_ecg: np.ndarray,
    classifier: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
):
    """KL(P_real || P_recon) and top-1 flip rate over diagnostic superclass probabilities.

    Mirrors compute_bce()'s structure (same idea: consistency of a frozen
    classifier's output between real and reconstructed ECG) but the classifier
    here is trained on real PTB-XL diagnoses rather than CLEF's label-free
    pretraining objective -- the question Guan et al. asked of ECGFounder.
    Lower is better for both.
    """
    eps = 1e-7
    kl_vals, flips = [], []
    N = len(true_ecg)
    for start in range(0, N, batch_size):
        t = torch.from_numpy(true_ecg[start:start + batch_size].astype(np.float32)).unsqueeze(1).to(device)
        p = torch.from_numpy(pred_ecg[start:start + batch_size].astype(np.float32)).unsqueeze(1).to(device)
        with torch.no_grad():
            p_real  = classifier(t).cpu().numpy()
            p_recon = classifier(p).cpu().numpy()
        pr = np.clip(p_real,  eps, 1.0 - eps)
        pc = np.clip(p_recon, eps, 1.0 - eps)
        kl = (pr * np.log(pr / pc) + (1.0 - pr) * np.log((1.0 - pr) / (1.0 - pc))).sum(axis=1)
        kl_vals.extend(kl.tolist())
        flips.extend((np.argmax(p_real, axis=1) != np.argmax(p_recon, axis=1)).tolist())
    return float(np.mean(kl_vals)), float(np.mean(flips))


def _collect_predictions(model, loader, device):
    all_true, all_pred = [], []
    model.eval()
    with torch.no_grad():
        for ppg, ecg in loader:
            pred = model(ppg.to(device))
            all_true.append(ecg.squeeze(-1).cpu().numpy())
            all_pred.append(pred.squeeze(-1).cpu().numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


# ---------------------------------------------------------------------------
# Checkpoint discovery: group *_fold_NN_best.pt files by (variant_label, arch)
# ---------------------------------------------------------------------------

def _discover_checkpoints(pattern: str):
    runs = defaultdict(list)
    for path in glob.glob(pattern, recursive=True):
        m = _CKPT_RE.search(os.path.basename(path))
        if not m:
            continue
        arch = m.group("arch")
        fold = int(m.group("fold"))
        # compare_reheartnet.py keys checkpoints by loss-variant subdirectory
        # (checkpoints/<model_key>/reheartnet_fold_NN_best.pt); run_cv.py writes
        # straight into checkpoints/. Prefer the subdirectory name when present.
        parent = os.path.basename(os.path.dirname(path))
        label = parent if parent not in ("checkpoints", "") else arch
        runs[(label, arch)].append((fold, path))
    return runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc diagnostic-consistency eval (additive to BCE)")
    parser.add_argument("--diagnostic-classifier", type=str, required=True,
                        help="Path to checkpoints/ptbxl_diagnostic_classifier.pt "
                             "(scripts/train_diagnostic_classifier.py output)")
    parser.add_argument("--checkpoint-glob", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "**", "*_fold_*_best.pt"),
                        help="Recursive glob matching saved fold checkpoints "
                             "(default covers both run_cv.py and compare_reheartnet.py layouts)")
    parser.add_argument("--clef-path", type=str, default=None,
                        help="CLEF encoder checkpoint (required for clef_probe classifiers).")
    parser.add_argument("--clef-dir", type=str, default=config.CLEF_CHECKPOINT_DIR)
    parser.add_argument("--clef-size", type=str, default="medium",
                        choices=["small", "medium", "large"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--out", type=str, default=os.path.join("results", "diag_consistency.json"))
    args = parser.parse_args()

    device = config.DEVICE
    print(f"Device: {device}")

    # Peek at checkpoint type to decide whether CLEF encoder is needed
    _peek = torch.load(args.diagnostic_classifier, map_location="cpu")
    clef_encoder = None
    if _peek.get("type") == "clef_probe":
        clef_path = args.clef_path or os.path.join(args.clef_dir, f"clef_{args.clef_size}.ckpt")
        print(f"Loading CLEF encoder ({args.clef_size}) from {clef_path} ...")
        clef_encoder = load_clef_encoder(clef_path, args.clef_size, device)

    classifier, superclasses = _load_diagnostic_classifier(
        args.diagnostic_classifier, clef_encoder=clef_encoder)
    classifier = classifier.to(device)
    print(f"Loaded diagnostic classifier ({args.diagnostic_classifier}, superclasses={superclasses})")

    runs = _discover_checkpoints(args.checkpoint_glob)
    if not runs:
        print(f"No checkpoints matched: {args.checkpoint_glob}")
        return
    print(f"Found {len(runs)} model variant(s): {[label for label, _ in sorted(runs)]}")

    batch_size = args.batch_size or config.BATCH_SIZE
    results = {}
    for (label, arch), fold_files in sorted(runs.items()):
        print(f"\n{'=' * 55}\n{label}  (architecture: {arch})\n{'=' * 55}")
        per_fold = []
        for fold_idx, ckpt_path in sorted(fold_files):
            ckpt = torch.load(ckpt_path, map_location=device)
            test_subs   = ckpt.get("fold_subjects")
            hidden_size = ckpt.get("hidden_size", config.HIDDEN_SIZE)
            if not test_subs:
                print(f"  [fold {fold_idx:02d}] checkpoint missing 'fold_subjects' -- skipping ({ckpt_path})")
                continue

            model = get_model(arch, hidden_size=hidden_size).to(device)
            model.load_state_dict(ckpt["model_state_dict"])

            test_ds     = build_test_dataset(test_subs)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

            true_arr, pred_arr = _collect_predictions(model, test_loader, device)
            kl, flip_rate = _diag_consistency(true_arr, pred_arr, classifier, device)
            per_fold.append({
                "fold": fold_idx, "subjects": test_subs,
                "diag_kl": kl, "diag_flip_rate": flip_rate,
            })
            print(f"  [fold {fold_idx:02d}] diag_kl={kl:.4f}  diag_flip_rate={flip_rate:.3f}  "
                  f"(n={len(true_arr)} windows, test={test_subs})")

        if not per_fold:
            continue

        kl_mean, kl_lo, kl_hi = compute_confidence_interval([f["diag_kl"]        for f in per_fold])
        fr_mean, fr_lo, fr_hi = compute_confidence_interval([f["diag_flip_rate"] for f in per_fold])
        results[label] = {
            "architecture":   arch,
            "diag_kl":        {"mean": kl_mean, "ci95_low": kl_lo, "ci95_high": kl_hi},
            "diag_flip_rate": {"mean": fr_mean, "ci95_low": fr_lo, "ci95_high": fr_hi},
            "per_fold":       per_fold,
        }
        print(f"  -> mean diag_kl={kl_mean:.4f} [{kl_lo:.4f}, {kl_hi:.4f}]   "
              f"mean diag_flip_rate={fr_mean:.3f} [{fr_lo:.3f}, {fr_hi:.3f}]")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved diagnostic-consistency results to {args.out}")
    print("These diag_kl / diag_flip_rate values are additive: merge them into your "
          "existing rmse/prd/bce/emd/... comparison tables alongside the unaffected "
          "CLEF-based BCE -- no need to relaunch any reconstruction training run.")


if __name__ == "__main__":
    main()
