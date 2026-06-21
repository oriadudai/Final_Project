"""Breaks down phase_align_ppg's effect into the 'wrapped' (np.roll
wrap-around) segment vs. the 'non-wrapped' (correctly-shifted) segment of
each aligned window, to determine whether the net correlation drop found by
diag_phase_align_quality.py is driven by the wrap-around artifact or by the
lag search itself finding unhelpful shifts.

No model/GPU needed -- pure signal-processing diagnostic on raw BIDMC data.

Usage:
    python scripts/diag_phase_align_wrap_breakdown.py --window-sec 4.0 --overlap-frac 0.0 --bandpass
    python scripts/diag_phase_align_wrap_breakdown.py --window-sec 10 --overlap-frac 0.5 --no-bandpass
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import build_subject_windows, get_all_record_names


def find_lag(ppg_w: np.ndarray, ecg_w: np.ndarray, max_lag: int = 125) -> int:
    N = len(ecg_w)
    ecg_c = ecg_w - ecg_w.mean()
    ppg_c = ppg_w - ppg_w.mean()
    full_corr = np.correlate(ecg_c, ppg_c, mode="full")
    center = N - 1
    window = full_corr[center - max_lag: center + max_lag + 1]
    return int(np.argmax(window) - max_lag)


def safe_abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(abs(np.corrcoef(a, b)[0, 1]))


def split_segments(N: int, lag: int):
    """Indices of the (wrapped, non_wrapped) segments of np.roll(x, -lag)."""
    L = abs(lag)
    if lag > 0:
        return np.arange(N - L, N), np.arange(0, N - L)
    return np.arange(0, L), np.arange(L, N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=53)
    ap.add_argument("--window-sec", type=float, default=4.0)
    ap.add_argument("--overlap-frac", type=float, default=0.0)
    ap.add_argument("--bandpass", dest="bandpass", action="store_true", default=True)
    ap.add_argument("--no-bandpass", dest="bandpass", action="store_false")
    ap.add_argument("--max-lag", type=int, default=125)
    ap.add_argument("--min-lag", type=int, default=10,
                    help="skip windows with |lag| below this (segment too short for stable corr)")
    args = ap.parse_args()

    subjects = get_all_record_names()[: args.n_subjects]

    wrap_aligned, wrap_baseline = [], []
    nonwrap_aligned, nonwrap_baseline = [], []
    n_total, n_qualifying = 0, 0

    for subj in subjects:
        ppg_un, ecg = build_subject_windows(
            subj, apply_phase_align=False,
            window_sec=args.window_sec, overlap_frac=args.overlap_frac,
            apply_bandpass=args.bandpass,
        )
        for ppg_w, ecg_w in zip(ppg_un, ecg):
            n_total += 1
            lag = find_lag(ppg_w, ecg_w, max_lag=args.max_lag)
            if abs(lag) < args.min_lag:
                continue
            n_qualifying += 1
            ppg_aligned = np.roll(ppg_w, -lag)
            wrap_idx, nonwrap_idx = split_segments(len(ppg_w), lag)

            wrap_aligned.append(safe_abs_corr(ppg_aligned[wrap_idx], ecg_w[wrap_idx]))
            wrap_baseline.append(safe_abs_corr(ppg_w[wrap_idx], ecg_w[wrap_idx]))
            nonwrap_aligned.append(safe_abs_corr(ppg_aligned[nonwrap_idx], ecg_w[nonwrap_idx]))
            nonwrap_baseline.append(safe_abs_corr(ppg_w[nonwrap_idx], ecg_w[nonwrap_idx]))

    def stats(name, aligned, baseline):
        a, b = np.array(aligned), np.array(baseline)
        valid = ~np.isnan(a) & ~np.isnan(b)
        a, b = a[valid], b[valid]
        print(f"  {name}:")
        print(f"    baseline mean |corr| = {b.mean():.4f}   aligned mean |corr| = {a.mean():.4f}"
              f"   delta = {a.mean() - b.mean():+.4f}")
        print(f"    aligned > baseline in {np.mean(a > b) * 100:.1f}% of windows  (n={len(a)})")

    print("=== Wrap-around vs non-wrapped segment breakdown ===")
    print(f"window_sec={args.window_sec}  overlap_frac={args.overlap_frac}  "
          f"bandpass={args.bandpass}  max_lag={args.max_lag}  min_lag={args.min_lag}")
    print(f"windows total={n_total}  qualifying (|lag|>={args.min_lag})="
          f"{n_qualifying} ({n_qualifying / n_total * 100:.1f}%)")
    print()
    stats("non-wrapped segment (correctly shifted region)", nonwrap_aligned, nonwrap_baseline)
    print()
    stats("wrapped segment (np.roll wrap-around artifact)", wrap_aligned, wrap_baseline)


if __name__ == "__main__":
    main()
