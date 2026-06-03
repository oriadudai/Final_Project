import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from core.config import DATA_DIR, BATCH_SIZE, SEQ_LEN

class BIDMCDataset(Dataset):
    """
    Custom Dataset to load and slice raw 8-minute PPG/ECG signals 
    from the BIDMC .mat database file into fixed windows.
    """
    def __init__(self, mat_file_path, subject_ids):
        """
        mat_file_path: נתיב פיזי לקובץ ה-mat של הנתונים
        subject_ids: רשימת אינדקסים/מטופלים שנרצה לטעון (למשל עבור אימון או טסט)
        """
        super().__init__()
        
        # טעינת קובץ המטלאב המלא לזיכרון
        print(f"Loading matrix data from {mat_file_path}...")
        mat_contents = sio.load_loadmat(mat_file_path)
        raw_data = mat_contents['data'][0] # שליפת המערך המרכזי שנקרא data
        
        self.ppg_windows = []
        self.ecg_windows = []
        
        # לולאה שעוברת רק על המטופלים שנבחרו לצורך חלוקת הנתונים
        for sub_idx in subject_ids:
            # שליפת אות ה-PPG וה-ECG באורך מלא (8 דקות = 60,000 דגימות בתדר 125Hz)
            # לפי המבנה הרשמי שמופיע בקובץ ה-README שצירפת
            ppg_signal = raw_data[sub_idx]['ppg'][0][0]['v'].flatten()
            ecg_signal = raw_data[sub_idx]['ekg'][0][0]['v'].flatten()
            
            # חישוב כמה חלונות באורך 1250 דגימות (10 שניות) נכנסים באות המלא
            num_windows = len(ppg_signal) // SEQ_LEN
            
            for i in range(num_windows):
                start = i * SEQ_LEN
                end = start + SEQ_LEN
                
                # חיתוך המקטע הספציפי מהאות
                ppg_seg = ppg_signal[start:end]
                ecg_seg = ecg_signal[start:end]
                
                # בדיקה מהירה שהמקטע מלא ולא פגום
                if len(ppg_seg) == SEQ_LEN and len(ecg_seg) == SEQ_LEN:
                    # ביצוע נרמול בסיסי מסוג Z-score (ממוצע 0, סטיית תקן 1)
                    # כדי לעזור למודל להתכנס מהר יותר באימון
                    ppg_norm = (ppg_seg - np.mean(ppg_seg)) / (np.std(ppg_seg) + 1e-8)
                    ecg_norm = (ecg_seg - np.mean(ecg_seg)) / (np.std(ecg_seg) + 1e-8)
                    
                    self.ppg_windows.append(ppg_norm)
                    self.ecg_windows.append(ecg_norm)
                    
        print(f"Ingestion complete. Total extracted training/testing slices: {len(self.ppg_windows)}")

    def __len__(self):
        # מחזיר את כמות חלונות הזמן הכוללת שחילצנו
        return len(self.ppg_windows)

    def __getitem__(self, idx):
        # שליפת החלון המבוקש והפיכתו לטנזור PyTorch בצורה הנדרשת למודל
        # Shape expected by model: (seq_len, 1) -> (1250, 1)
        ppg_tensor = torch.tensor(self.ppg_windows[idx], dtype=torch.float32).unsqueeze(-1)
        ecg_tensor = torch.tensor(self.ecg_windows[idx], dtype=torch.float32).unsqueeze(-1)
        
        return ppg_tensor, ecg_tensor


def get_loaders():
    """
    Helper function to initialize Datasets and return PyTorch DataLoaders
    using Subject-level split (Patients 0-40 for Train, 41-52 for Test).
    """
    # הגדרת נתיב לקובץ הפיזי במערכת
    mat_path = os.path.join(DATA_DIR, "bidmc_data.mat")
    
    # חלוקה מבוססת נבדקים (Subject-level split) כדי למנוע זליגת מידע רפואי
    train_subjects = list(range(0, 40))  # 40 מטופלים ראשונים לאימון
    test_subjects = list(range(40, 53))   # 13 המטופלים הנותרים לבדיקה סופית
    
    print("--- Creating Train Dataset ---")
    train_dataset = BIDMCDataset(mat_path, train_subjects)
    
    print("--- Creating Test Dataset ---")
    test_dataset = BIDMCDataset(mat_path, test_subjects)
    
    # בניית ה-DataLoader של PyTorch שיזין את ה-Batches למודל
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader


if __name__ == "__main__":
    print("=== Testing BIDMC Data Loader Ingestion ===")
    try:
        train_l, test_l = get_loaders()
        # שליפת Batch בודד לדוגמה כדי לוודא שהממדים תקינים
        sample_ppg, sample_ecg = next(iter(train_l))
        print(f"Data Loader Test Passed Successfully!")
        print(f"Batch PPG Tensor Shape: {sample_ppg.shape}") # צפוי לקבל: [64, 1250, 1]
        print(f"Batch ECG Tensor Shape: {sample_ecg.shape}") # צפוי לקבל: [64, 1250, 1]
    except Exception as e:
        print(f"Data Loader Ingestion Failed: {str(e)}")
