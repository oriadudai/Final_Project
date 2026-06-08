"""Train a real diagnostic classifier on PTB-XL for the diag-consistency eval metric.

Background
----------
core/evaluate.py's BCE metric measures consistency in CLEF's clinically-*pretrained*
feature space (no labels needed). This script trains a complementary classifier on
*labelled* diagnoses (PTB-XL's 5 standard superclasses: NORM, MI, STTC, CD, HYP),
producing a frozen model that scripts/evaluate_diagnostic_consistency.py can
use to ask a sharper question: "does the reconstruction preserve enough information
for an actual pathology classifier to reach the same diagnosis as on the real ECG?"

This is a *one-time* training job — run once, save a checkpoint, then reuse it to
score every already-trained ReHeartNet variant/fold via --diagnostic-classifier. No
need to relaunch the reconstruction CV runs.

Why PTB-XL (not BIDMC): BIDMC has no diagnostic labels (it's an ICU PPG/respiration
dataset). PTB-XL provides clinician-validated multi-label diagnoses, but only for
ECG — there is no paired PPG, so the classifier cannot be trained end-to-end on the
reconstruction task. Training it standalone on real PTB-XL ECG, then applying it
(frozen) to both real and reconstructed BIDMC ECG, is what makes the metric additive
to — not a replacement of — the existing CLEF-based BCE.

Usage:
    python scripts/download_ptbxl.py                  # one-time download (~1.7 GB)
    python scripts/train_diagnostic_classifier.py     # one-time training (~hours on 1 GPU)

Output:
    checkpoints/ptbxl_diagnostic_classifier.pt
    → pass via --diagnostic-classifier to scripts/evaluate_diagnostic_consistency.py
"""

import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from scipy.signal import resample
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.config as config
from core.models.ptbxl_classifier import SurrogateECGClassifier
from src.preprocessing import normalize_signal

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


# ---------------------------------------------------------------------------
# Label mapping: SCP codes -> 5 diagnostic superclasses (multi-label)
# ---------------------------------------------------------------------------

def _build_scp_to_superclass(ptbxl_dir: str) -> dict:
    """Map each SCP statement code to its diagnostic superclass (PTB-XL metadata)."""
    scp_df = pd.read_csv(os.path.join(ptbxl_dir, "scp_statements.csv"), index_col=0)
    scp_df = scp_df[scp_df["diagnostic"] == 1]
    return scp_df["diagnostic_class"].dropna().to_dict()


def _labels_for_record(scp_codes_str: str, scp_to_super: dict) -> np.ndarray:
    """Parse a record's scp_codes dict-string into a 5-dim multi-hot label vector."""
    label = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    try:
        codes = ast.literal_eval(scp_codes_str)
    except (ValueError, SyntaxError):
        return label
    for code in codes:
        superclass = scp_to_super.get(code)
        if superclass in SUPERCLASSES:
            label[SUPERCLASSES.index(superclass)] = 1.0
    return label


# ---------------------------------------------------------------------------
# Dataset: Lead II @ 125 Hz, 10 s windows, z-scored — matches ReHeartNet's
# reconstruction output format so the classifier can later score it directly.
# ---------------------------------------------------------------------------

class PTBXLDiagnosticDataset(Dataset):
    def __init__(self, ptbxl_dir: str, df: pd.DataFrame, scp_to_super: dict):
        self.ptbxl_dir = ptbxl_dir
        self.records = df["filename_lr"].tolist()       # 100 Hz records
        self.labels = np.stack([
            _labels_for_record(codes, scp_to_super) for codes in df["scp_codes"]
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rel_path = self.records[idx].replace(".dat", "").replace(".hea", "")
        record = wfdb.rdrecord(os.path.join(self.ptbxl_dir, rel_path))
        sig_names = [s.strip().upper() for s in record.sig_name]
        lead_ii = record.p_signal[:, sig_names.index("II")]   # 100 Hz, 10 s -> 1000 samples

        resampled = resample(lead_ii, config.SEQ_LEN)         # -> 125 Hz, 1250 samples
        windowed = normalize_signal(resampled).astype(np.float32)

        x = torch.from_numpy(windowed).unsqueeze(0)           # (1, SEQ_LEN)
        y = torch.from_numpy(self.labels[idx])                # (5,)
        return x, y


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_batches = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PTB-XL diagnostic classifier (one-time)")
    parser.add_argument("--ptbxl-dir", type=str, default=os.path.join("data", "PTBXL"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "ptbxl_diagnostic_classifier.pt"))
    args = parser.parse_args()

    device = config.DEVICE
    print(f"Device: {device}")

    db = pd.read_csv(os.path.join(args.ptbxl_dir, "ptbxl_database.csv"), index_col="ecg_id")
    scp_to_super = _build_scp_to_superclass(args.ptbxl_dir)

    # Official PTB-XL split: folds 1-8 train, 9 val, 10 test (Wagner et al. protocol)
    train_df = db[db["strat_fold"] <= 8]
    val_df   = db[db["strat_fold"] == 9]
    print(f"Train records: {len(train_df)} | Val records: {len(val_df)}")

    train_ds = PTBXLDiagnosticDataset(args.ptbxl_dir, train_df, scp_to_super)
    val_ds   = PTBXLDiagnosticDataset(args.ptbxl_dir, val_df,   scp_to_super)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = SurrogateECGClassifier(num_classes=len(SUPERCLASSES)).to(device)
    for p in model.parameters():
        p.requires_grad = True
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss   = _run_epoch(model, val_loader,   device, criterion, optimizer=None)
        print(f"[Epoch {epoch:02d}/{args.epochs}] train_bce={train_loss:.4f}  val_bce={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "state_dict":   model.state_dict(),
                "superclasses": SUPERCLASSES,
                "val_bce":      val_loss,
                "epoch":        epoch,
            }, args.out)
            print(f"  -> saved new best checkpoint to {args.out} (val_bce={val_loss:.4f})")

    print(f"\nDone. Best val BCE = {best_val:.4f}")
    print(f"Use it in evaluation via:  --diagnostic-classifier {args.out}")


if __name__ == "__main__":
    main()
