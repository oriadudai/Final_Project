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




# LOSO Cross-Validation
import random

# הגדרת subjects
record_names = [f"bidmc{str(i).zfill(2)}" for i in range(1, 54)]

# הפרשת Validation subjects (לפני ה-LOSO)
random.seed(42)
val_subjects = random.sample(record_names, 3)  # 2-3 subjects קבועים
loso_subjects = [s for s in record_names if s not in val_subjects]

print(f"Validation subjects (for hyperparameter tuning): {val_subjects}")
print(f"LOSO subjects: {len(loso_subjects)}")

# Grid Search על ה-Validation subjects
delta_options = [0.1, 0.5, 1.0]
lambda_options = [0.1, 0.5, 1.0]

val_dataset = build_full_dataset(val_subjects)

best_params = None
best_val_loss = float("inf")

for delta in delta_options:
    for lam in lambda_options:

        print(f"\n--- Grid Search: delta={delta}, lambda={lam} ---")

        # כאן תאמן מודל קטן על שאר ה-val subjects ותעריך על אחד מהם
        # לדוגמה: train על 2, test על 1 מתוך ה-val_subjects
        val_train = build_full_dataset(val_subjects[:-1])
        val_test  = build_full_dataset([val_subjects[-1]])

        # TODO: אימון מודל עם delta ו-lambda הנוכחיים
        # val_loss = train_and_evaluate(val_train, val_test, delta, lam)

        val_loss = 0.0  # placeholder — תחליפי בתוצאה האמיתית

        print(f"Validation loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {"delta": delta, "lambda": lam}

print(f"\n✅ Best hyperparameters: {best_params}")

# LOSO על שאר ה-subjects עם הפרמטרים הטובים
best_delta  = best_params["delta"]
best_lambda = best_params["lambda"]

for test_subject in loso_subjects:

    print(f"\n===== LOSO Fold: Testing on {test_subject} =====")

    train_subjects = [s for s in loso_subjects if s != test_subject]

    train_dataset = build_full_dataset(train_subjects)
    test_dataset  = build_full_dataset([test_subject])

    print(f"Train size: {len(train_dataset)} | Test size: {len(test_dataset)}")

    # TODO: אימון עם best_delta ו-best_lambda
    # model = train_model(train_dataset, best_delta, best_lambda)
    # results = evaluate_model(model, test_dataset)