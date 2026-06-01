import os
import numpy as np
import wfdb
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import core.config as config


def load_bidmc_record(record_name):
    """Load ECG (Lead II) and PPG (PLETH) from a BIDMC WFDB record.

    Both channels come from the same synchronized p_signal matrix.
    Channel order varies per subject, so we look up by name.
    The *n variants (bidmcXXn) are numerics files and must NOT be passed here.
    """
    path = os.path.join(config.DATA_DIR, "BIDMC", record_name)
    record = wfdb.rdrecord(path)
    sig_names = [s.strip().rstrip(",").upper() for s in record.sig_name]
    assert "PLETH" in sig_names and "II" in sig_names, (
        f"Missing channels in {record_name}: {sig_names}"
    )
    ecg = record.p_signal[:, sig_names.index("II")]
    ppg = record.p_signal[:, sig_names.index("PLETH")]
    return ecg, ppg, record.fs


def normalize_signal(signal):
    """Z-score normalization over the full signal array."""
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-8:
        return signal - mean
    return (signal - mean) / std


def phase_align_ppg(ppg, ecg, max_lag=125):
    """Shift PPG to reduce systematic phase offset relative to ECG.

    Finds the lag that maximises cross-correlation within ±max_lag samples
    (default ±1 second at 125 Hz) then rolls the PPG by that offset.
    Applied to training windows only; never applied at test time.
    """
    N = len(ecg)
    ecg_c = ecg - ecg.mean()
    ppg_c = ppg - ppg.mean()
    full_corr = np.correlate(ecg_c, ppg_c, mode="full")
    center = N - 1
    window = full_corr[center - max_lag : center + max_lag + 1]
    lag = np.argmax(window) - max_lag
    return np.roll(ppg, -lag)


def create_windows(signal, window_size, step_size):
    """Slide a window across signal and return a stacked array of windows."""
    starts = range(0, len(signal) - window_size + 1, step_size)
    windows = [signal[s : s + window_size] for s in starts]
    return np.array(windows, dtype=np.float32)


def build_subject_windows(record_name, apply_phase_align=False):
    """Load, normalize, and window ECG+PPG for one BIDMC subject.

    Returns:
        ppg_windows: np.ndarray of shape (N_windows, config.SEQ_LEN)
        ecg_windows: np.ndarray of shape (N_windows, config.SEQ_LEN)
    """
    ecg_raw, ppg_raw, fs = load_bidmc_record(record_name)

    ecg_norm = normalize_signal(ecg_raw)
    ppg_norm = normalize_signal(ppg_raw)

    window_size = config.SEQ_LEN          # 1250 samples = 10 s @ 125 Hz
    step_size   = config.SEQ_LEN // 2    # 625 samples = 50 % overlap

    ecg_windows = create_windows(ecg_norm, window_size, step_size)
    ppg_windows = create_windows(ppg_norm, window_size, step_size)

    if apply_phase_align:
        aligned = []
        for ppg_w, ecg_w in zip(ppg_windows, ecg_windows):
            aligned.append(phase_align_ppg(ppg_w, ecg_w))
        ppg_windows = np.stack(aligned, axis=0)

    return ppg_windows, ecg_windows


def get_all_record_names():
    """Return the list of the 53 main BIDMC record names."""
    return [f"bidmc{i:02d}" for i in range(1, 54)]
