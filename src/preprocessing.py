import wfdb
import numpy as np
from scipy.signal import resample


# טעינת רשומת BIDMC
def load_bidmc_record(record_name):

    path = f"C:/Users/aslan/OneDrive/Desktop/PPG2ECG_project/data/BIDMC/{record_name}"

    record = wfdb.rdrecord(path)

    signals = record.p_signal

    # חילוץ אותות ECG ו-PPG
    ecg = signals[:, 0]
    ppg = signals[:, 1]

    return ecg, ppg


# ביצוע Resampling לאות
def resample_signal(signal, old_fs=125, new_fs=250):

    new_length = int(len(signal) * new_fs / old_fs)

    return resample(signal, new_length)


# Normalization של האות
def normalize_signal(signal):

    mean = np.mean(signal)
    std = np.std(signal)

    normalized_signal = (signal - mean) / std

    return normalized_signal


# חלוקת האות ל-segments
def create_segments(signal, window_size, step_size):

    segments = []

    for start in range(0, len(signal) - window_size, step_size):

        end = start + window_size

        segment = signal[start:end]

        segments.append(segment)

    return np.array(segments)


# בניית dataset עבור subject אחד
def build_subject_dataset(record_name):

    # טעינת אותות
    ecg, ppg = load_bidmc_record(record_name)

    # התאמת תדר דגימה
    ecg = resample_signal(ecg)
    ppg = resample_signal(ppg)

    # Normalization
    ecg = normalize_signal(ecg)
    ppg = normalize_signal(ppg)

    # הגדרות segmentation
    fs = 250

    window_size = 10 * fs
    step_size = 5 * fs

    # יצירת segments
    ecg_segments = create_segments(
        ecg,
        window_size,
        step_size
    )

    ppg_segments = create_segments(
        ppg,
        window_size,
        step_size
    )

    # יצירת dataset
    dataset = []

    for i in range(len(ecg_segments)):

        sample = {
            "ppg": ppg_segments[i],
            "ecg": ecg_segments[i],
            "subject_id": record_name
        }

        dataset.append(sample)

    return dataset


# בניית dataset ממספר subjects
def build_full_dataset(record_names):

    full_dataset = []

    for record_name in record_names:

        print(f"Loading {record_name}...")

        subject_dataset = build_subject_dataset(record_name)

        full_dataset.extend(subject_dataset)

    return full_dataset



# רשימת subjects
record_names = []

for i in range(1, 54):

    record_name = f"bidmc{str(i).zfill(2)}"

    record_names.append(record_name)

# LOSO Cross-Validation
for test_subject in record_names:

    print(f"\n===== Testing on {test_subject} =====")

    # subject לבדיקה
    test_subjects = [test_subject]

    # כל שאר ה-subjects לאימון
    train_subjects = [
        subject for subject in record_names
        if subject != test_subject
    ]

    print("Train subjects:", len(train_subjects))
    print("Test subject:", test_subject)

    # בניית datasets
    train_dataset = build_full_dataset(train_subjects)

    test_dataset = build_full_dataset(test_subjects)

    # מידע
    print("\nTrain dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    # בדיקת sample
    print("\nFirst sample keys:")
    print(train_dataset[0].keys())

    print("\nPPG shape:")
    print(train_dataset[0]["ppg"].shape)

    print("\nECG shape:")
    print(train_dataset[0]["ecg"].shape)