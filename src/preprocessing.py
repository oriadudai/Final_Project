import os
import numpy as np
import wfdb
import sys
from scipy.signal import firwin, lfilter

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


def bandpass_fir(signal: np.ndarray, low_hz: float, high_hz: float, fs: float,
                 numtaps: int = 127) -> np.ndarray:
    """Type-I linear-phase FIR bandpass filter (constant group delay across spectrum).

    Matches the filter described in Lee et al.: ECG 0.5–55 Hz, PPG 0.5–10 Hz.
    numtaps=127 → group delay of 63 samples; both signals share the same delay,
    so relative ECG/PPG alignment is preserved.
    """
    nyq = fs / 2.0
    coeffs = firwin(numtaps, [low_hz / nyq, high_hz / nyq], pass_zero=False)
    return lfilter(coeffs, 1.0, signal)


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


def build_subject_windows(
    record_name,
    apply_phase_align: bool = False,
    window_sec: float = None,
    overlap_frac: float = 0.5,
    apply_bandpass: bool = False,
):
    """Load, normalize, filter, and window ECG+PPG for one BIDMC subject.

    Args:
        record_name:       BIDMC record identifier (e.g. "bidmc01").
        apply_phase_align: Cross-correlation phase alignment of PPG to ECG.
                           Apply to training windows only; never at test time.
        window_sec:        Window length in seconds.  None → config.SEQ_LEN
                           (default 10 s = 1250 samples @ 125 Hz).
                           Pass 4.0 to match the original Lee et al. setup.
        overlap_frac:      Fraction of window overlap between consecutive windows.
                           0.0 = non-overlapping (original paper), 0.5 = 50% (our default).
        apply_bandpass:    If True, apply type-I FIR bandpass before windowing:
                           ECG 0.5–55 Hz, PPG 0.5–10 Hz (matches Lee et al.).

    Returns:
        ppg_windows: np.ndarray of shape (N_windows, window_size)
        ecg_windows: np.ndarray of shape (N_windows, window_size)
    """
    ecg_raw, ppg_raw, fs = load_bidmc_record(record_name)

    ecg_norm = normalize_signal(ecg_raw)
    ppg_norm = normalize_signal(ppg_raw)

    if apply_bandpass:
        ecg_norm = bandpass_fir(ecg_norm, 0.5, 55.0, fs)
        ppg_norm = bandpass_fir(ppg_norm, 0.5, 10.0, fs)

    window_size = int(round(window_sec * fs)) if window_sec is not None else config.SEQ_LEN
    step_size   = max(1, int(round(window_size * (1.0 - overlap_frac))))

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
