import os
import json
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import KFold

from src.preprocessing import build_subject_windows


class BIDMCDataset(Dataset):
    """PyTorch Dataset wrapping PPG and ECG window arrays for BIDMC subjects."""

    def __init__(self, ppg_windows: np.ndarray, ecg_windows: np.ndarray):
        # Shape: (N, SEQ_LEN, 1) — channel dim appended for model compatibility
        self.ppg = torch.from_numpy(ppg_windows.astype(np.float32)).unsqueeze(-1)
        self.ecg = torch.from_numpy(ecg_windows.astype(np.float32)).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.ppg)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.ppg[idx], self.ecg[idx]


def _load_subjects(
    subjects: List[str],
    apply_align: bool,
    window_sec: float = None,
    overlap_frac: float = 0.5,
    apply_bandpass: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate windows from multiple subjects into single arrays."""
    all_ppg, all_ecg = [], []
    for subj in subjects:
        ppg_w, ecg_w = build_subject_windows(
            subj,
            apply_phase_align=apply_align,
            window_sec=window_sec,
            overlap_frac=overlap_frac,
            apply_bandpass=apply_bandpass,
        )
        all_ppg.append(ppg_w)
        all_ecg.append(ecg_w)
    return np.concatenate(all_ppg, axis=0), np.concatenate(all_ecg, axis=0)


def build_group_fold(
    train_subjects: List[str],
    test_subjects: List[str],
    apply_align: bool = True,
    window_sec: float = None,
    overlap_frac: float = 0.5,
    apply_bandpass: bool = False,
) -> Tuple[BIDMCDataset, BIDMCDataset]:
    """Build train and test datasets for one CV fold.

    Phase alignment is applied to training subjects only; test data is never
    aligned (alignment would require ECG at inference time, which defeats the
    purpose of the reconstruction task).

    Args:
        window_sec:    Window length in seconds (None → config default 10 s).
        overlap_frac:  Window overlap fraction (0.0 = non-overlapping).
        apply_bandpass: Apply FIR bandpass pre-filter (ECG 0.5–55 Hz, PPG 0.5–10 Hz).
    """
    train_ppg, train_ecg = _load_subjects(
        train_subjects, apply_align=apply_align,
        window_sec=window_sec, overlap_frac=overlap_frac, apply_bandpass=apply_bandpass,
    )
    test_ppg, test_ecg = _load_subjects(
        test_subjects, apply_align=False,
        window_sec=window_sec, overlap_frac=overlap_frac, apply_bandpass=apply_bandpass,
    )
    return BIDMCDataset(train_ppg, train_ecg), BIDMCDataset(test_ppg, test_ecg)


def get_cv_splits(
    all_subjects: List[str],
    n_splits: int = 8,
    random_state: int = 42,
    save_path: str = "results/fold_assignments.json",
) -> List[Tuple[List[str], List[str]]]:
    """Split subject list into n_splits folds (~7 subjects per test fold for 53 subjects).

    Saves the assignments to save_path for reproducibility.
    Returns list of (train_subjects, test_subjects) tuples.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    subjects_arr = np.array(all_subjects)

    splits = []
    assignments = {}
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(subjects_arr)):
        train_subs = subjects_arr[train_idx].tolist()
        test_subs  = subjects_arr[test_idx].tolist()
        splits.append((train_subs, test_subs))
        assignments[f"fold_{fold_idx:02d}"] = {
            "train": train_subs,
            "test":  test_subs,
        }

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(assignments, f, indent=2)

    return splits


def build_test_dataset(
    test_subjects: List[str],
    window_sec: float = None,
    overlap_frac: float = 0.5,
    apply_bandpass: bool = False,
) -> BIDMCDataset:
    """Build a test-only dataset without requiring a paired training set."""
    test_ppg, test_ecg = _load_subjects(
        test_subjects, apply_align=False,
        window_sec=window_sec, overlap_frac=overlap_frac,
        apply_bandpass=apply_bandpass,
    )
    return BIDMCDataset(test_ppg, test_ecg)


def get_loso_splits(
    all_subjects: List[str],
    save_path: str = "results/loso_assignments.json",
) -> List[Tuple[List[str], List[str]]]:
    """Leave-One-Subject-Out splits: each fold holds out exactly one subject.

    For BIDMC (53 subjects) this produces 53 folds, giving an unbiased
    subject-level evaluation with maximum training data per fold.

    Returns list of (train_subjects, [test_subject]) tuples.
    """
    splits = []
    assignments = {}
    for fold_idx, test_subj in enumerate(all_subjects):
        train_subs = [s for s in all_subjects if s != test_subj]
        splits.append((train_subs, [test_subj]))
        assignments[f"fold_{fold_idx:02d}"] = {
            "train": train_subs,
            "test":  [test_subj],
        }

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(assignments, f, indent=2)

    return splits


def split_train_val(
    dataset: BIDMCDataset,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[BIDMCDataset, BIDMCDataset]:
    """Randomly split a BIDMCDataset into train and validation subsets."""
    n_val   = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=generator)
