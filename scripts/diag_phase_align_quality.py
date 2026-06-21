"""Diagnose phase_align_ppg (src/preprocessing.py): does the per-window
cross-correlation alignment actually improve PPG-ECG correlation, or does
it lock onto cycle-shifted lags / introduce wrap-around discontinuities?

Pure signal-processing diagnostic on raw BIDMC data -- no model, no GPU,
runs in seconds.

Usage:
    # reheartnet_original / reheartnet_huber preprocessing (4s windows, no overlap, bandpass)
    python scripts/diag_phase_align_quality.py --window-sec 4.0 --overlap-frac 0.0 --bandpass

    # reheartnet_clef preprocessing (10s windows, 50% overlap, no bandpass)
    python scripts/diag_phase_align_quality.py --window-sec 10 --overlap-frac 0.5 --no-bandpass
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import build_subject_windows, get_all_record_names


def find_lag(ppg_w: np.ndarray, ecg_w: np.ndarray, max_lag: int = 125) -> int:
    """Replicates phase_align_ppg's lag search, returning the lag instead of
    applying np.roll."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=53)
    ap.add_argument("--window-sec", type=float, default=4.0)
    ap.add_argument("--overlap-frac", type=float, default=0.0)
    ap.add_argument("--bandpass", dest="bandpass", action="store_true", default=True)
    ap.add_argument("--no-bandpass", dest="bandpass", action="store_false")
    ap.add_argument("--max-lag", type=int, default=125)
    args = ap.parse_args()

    subjects = get_all_record_names()[: args.n_subjects]
    window_size = int(round(args.window_sec * 125)) if args.window_sec else 1250

    lags, corr_before, corr_after = [], [], []
    for subj in subjects:
        ppg_un, ecg = build_subject_windows(
            subj, apply_phase_align=False,
            window_sec=args.window_sec, overlap_frac=args.overlap_frac,
            apply_bandpass=args.bandpass,
        )
        for ppg_w, ecg_w in zip(ppg_un, ecg):
            lag = find_lag(ppg_w, ecg_w, max_lag=args.max_lag)
            ppg_aligned = np.roll(ppg_w, -lag)
            lags.append(lag)
            corr_before.append(safe_abs_corr(ppg_w, ecg_w))
            corr_after.append(safe_abs_corr(ppg_aligned, ecg_w))

    lags = np.array(lags)
    corr_before = np.array(corr_before)
    corr_after = np.array(corr_after)
    valid = ~np.isnan(corr_before) & ~np.isnan(corr_after)
    cb, ca = corr_before[valid], corr_after[valid]

    print("=== Phase-alignment diagnostic ===")
    print(f"window_sec={args.window_sec}  window_size={window_size} samples  "
          f"overlap_frac={args.overlap_frac}  bandpass={args.bandpass}  "
          f"max_lag={args.max_lag} ({args.max_lag / window_size:.0%} of window length)")
    print(f"subjects={len(subjects)}  windows={len(lags)}")
    print()
    print("--- lag distribution (samples) ---")
    print(f"  mean={lags.mean():.2f}  std={lags.std():.2f}  median={np.median(lags):.1f}")
    print(f"  |lag| >= {args.max_lag - 5} (saturating near boundary): "
          f"{np.mean(np.abs(lags) >= args.max_lag - 5) * 100:.1f}%")
    print(f"  |lag| >= {args.max_lag // 2}: "
          f"{np.mean(np.abs(lags) >= args.max_lag // 2) * 100:.1f}%")
    print(f"  |lag| <= 50 (plausible pulse-transit-time range): "
          f"{np.mean(np.abs(lags) <= 50) * 100:.1f}%")
    print()
    print("--- |corr(ppg, ecg)|: before vs after alignment ---")
    print(f"  mean before: {cb.mean():.4f}   median before: {np.median(cb):.4f}")
    print(f"  mean after:  {ca.mean():.4f}   median after:  {np.median(ca):.4f}")
    print(f"  alignment HURT  |corr| (after < before): {np.mean(ca < cb) * 100:.1f}%")
    print(f"  alignment HELPED |corr| (after > before): {np.mean(ca > cb) * 100:.1f}%")

    out_dir = "results/diag_phase_align"
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"w{args.window_sec}_o{args.overlap_frac}_bp{int(args.bandpass)}"
    out_path = os.path.join(out_dir, f"lag_corr_{suffix}.json")
    with open(out_path, "w") as f:
        json.dump({
            "lags": lags.tolist(),
            "corr_before": corr_before.tolist(),
            "corr_after": corr_after.tolist(),
        }, f)
    print(f"\nSaved raw arrays to {out_path}")


if __name__ == "__main__":
    main()
