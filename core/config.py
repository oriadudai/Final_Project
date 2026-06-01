import os
import torch

# =====================================================================
# 📂 הגדרות נתיבים (Paths)
# =====================================================================
# תיקיית המקור הכללית של הנתונים
DATA_DIR = "data"

# תיקייה לשמירת קבצי המשקולות וה-Checkpoints של המודל
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =====================================================================
# 📊 פרמטרים של האות הרפואי (Signal Parameters)
# =====================================================================
# תדר הדגימה של אותות ה-PPG וה-ECG 
FS = 125

# אורך כל חלון/מקטע זמן בשניות
WINDOW_SIZE = 10

# אורך וקטור הכניסה למודל בפועל (125 כפול 10 = 1250)
SEQ_LEN = FS * WINDOW_SIZE


# =====================================================================
# 🧠 היפר-פרמטרים של המודל (Model Hyperparameters)
# =====================================================================
# מספר ערוצי הקלט (ערוץ יחיד המייצג את אות ה-PPG)
INPUT_CHANNELS = 1

# מספר הנוירונים בשכבות ה-BiLSTM
HIDDEN_SIZE = 64

# מספר הבלוקים של ה-DC-BiLSTM
NUM_BLOCKS = 5


# =====================================================================
# 🚀 היפר-פרמטרים של תהליך האימון (Training Hyperparameters)
# =====================================================================
# גודל ה-Batch
BATCH_SIZE = 64

# קצב הלמידה ההתחלתי (Learning Rate)
LEARNING_RATE = 1e-3

# מספר מחזורי האימון
EPOCHS = 100

# הגדרת התקן העיבוד (GPU/CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# 🔍 הדפסת קונפיגורציה לבדיקה ראשונית
# =====================================================================
# =====================================================================
# 🏥 Clinical Loss Hyperparameters (eval_logic.pdf)
# =====================================================================
# Weight for the CLEF perceptual feature-matching loss term
LAMBDA_CLINICAL = 0.1

# Huber loss delta: controls L1/L2 transition point
HUBER_DELTA = 1.0


# =====================================================================
# 🧬 CLEF Foundation Model Settings
# =====================================================================
# Model size: "small" (256-dim, 5.5 MB), "medium" (1024-dim, 368 MB),
#             "large" (2048-dim, 3.6 GB)
CLEF_MODEL_SIZE = "small"

# Directory where CLEF checkpoint files are stored
CLEF_CHECKPOINT_DIR = "models/clef"


if __name__ == "__main__":
    print("=== ReHeartNet Configuration Initialized ===")
    print(f"Signal Sequence Length: {SEQ_LEN} samples")
    print(f"Training Environment Device: {DEVICE}")
    print(f"Lambda Clinical: {LAMBDA_CLINICAL}")
    print(f"Huber Delta: {HUBER_DELTA}")