import os
import torch
import torch.nn as nn
import numpy as np
# מייבאים את הגדרות המערכת והנתיבים
from core.config import DEVICE, SEQ_LEN, BATCH_SIZE, CHECKPOINT_DIR
from core.models.reheartnet import ReHeartNet

def compute_pearson_r(pred, target):
    """
    Computes the Pearson Correlation Coefficient (r) between reconstructed and ground truth signals.
    מחשבת את מקדם המתאם של פירסון בין האות המשוחזר לאות האמת.
    """
    # השטחת הטנזורים למערכים חד-ממדיים לצורך החישוב
    pred_flat = pred.detach().cpu().numpy().flatten()
    target_flat = target.detach().cpu().numpy().flatten()
    
    # חישוב מטריצת המתאם ושליפת הערך הרלוונטי
    corr_matrix = np.corrcoef(pred_flat, target_flat)
    return corr_matrix[0, 1]


def mock_pan_tompkins_beat_timing(pred_ecg, true_ecg):
    """
    Placeholder for the clinical Beat-Timing Error estimation using R-peak detection.
    סימולציה של מדד שגיאת תזמון הפעימות המבוסס על אלגוריתם פאן-טומפקינס.
    """
    # בשלב הבא, חבר הצוות שאחראי על המדדים (Workstream 4) יחליף שורות אלו 
    # בקריאה אמיתית ל- neurokit2.ecg_peaks() כדי למצוא את המיקום המדויק של ה-R-peaks.
    
    # כרגע, לצורך בדיקת שלמות הצינור, נחזיר שגיאת תזמון אקראית קטנה בשניות
    simulated_error_seconds = np.random.uniform(0.01, 0.05)
    return simulated_error_seconds


def evaluate_model(model, test_loader, device):
    """
    Runs inference on the test dataset and compiles medical and mathematical metrics.
    מריצה הסקה על נתוני הטסט ומחשבת את כל המדדים להערכת ביצועי הרשת.
    """
    model.eval()
    
    # שימוש ב-MSE כבסיס למדד ה-RMSE של המאמר
    mse_criterion = nn.MSELoss()
    
    total_mse = 0.0
    all_pearson_r = []
    all_beat_errors = []
    
    print("Running evaluation inference on test split batches...")
    
    # ביטול חישוב הגרדיאנטים כדי למנוע עומס על הזיכרון והמעבד
    with torch.no_grad():
        for ppg, ecg in test_loader:
            ppg, ecg = ppg.to(device), ecg.to(device)
            
            # הרצת אות ה-PPG במודל הקפוא לקבלת השחזור
            predictions = model(ppg)
            
            # 1. חישוב מדד השגיאה הריבועית
            loss_mse = mse_criterion(predictions, ecg)
            total_mse += loss_mse.item()
            
            # 2. חישוב מקדם המתאם של פירסון עבור ה-Batch הנוכחי
            r_val = compute_pearson_r(predictions, ecg)
            all_pearson_r.append(r_val)
            
            # 3. חישוב שגיאת תזמון הפעימות הקלינית
            beat_err = mock_pan_tompkins_beat_timing(predictions, ecg)
            all_beat_errors.append(beat_err)
            
    # חישוב הממוצעים הסופיים של המדדים
    avg_rmse = np.sqrt(total_mse / len(test_loader))
    avg_pearson_r = np.mean(all_pearson_r)
    avg_beat_error = np.mean(all_beat_errors)
    
    return avg_rmse, avg_pearson_r, avg_beat_error


def main():
    print("=== ReHeartNet Clinical Evaluation Pipeline ===")
    
    # 1. יצירת מופע של הרשת והעברתו למעבד
    model = ReHeartNet().to(DEVICE)
    
    # 2. ניסיון טעינה של מודל מאומן מתוך תיקיית ה-Checkpoints
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_reheartnet_model.pt")
    
    if os.path.exists(checkpoint_path):
        print(f"Loading trained weights from checkpoint: {checkpoint_path}")
        # במצב מקומי ללא GPU, אנו מוודאים שהמשקולות נפתחות ישירות על ה-CPU
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model weights loaded successfully.")
    else:
        print("⚠️ No trained checkpoint found. Running evaluation on randomly initialized weights.")
        
    # 3. יצירת נתוני מבחן מדומים (Mock Test Data) קלים ביותר לבדיקת שלמות הצינור
    # אנו מייצרים קבוצה בודדת (Batch) כדי לא להעמיס על המחשב שלך
    mock_test_loader = [
        (torch.randn(BATCH_SIZE, SEQ_LEN, 1), torch.randn(BATCH_SIZE, SEQ_LEN, 1))
    ]
    
    # 4. הרצת תהליך ההערכה המלא
    rmse, pearson_r, beat_timing_err = evaluate_model(model, mock_test_loader, DEVICE)
    
    # הדפסת טבלת המדדים הסופית באנגלית למסך הטרמינל
    print("\n==================================================")
    print("📊 FINAL QUANTITATIVE EVALUATION RESULTS")
    print("==================================================")
    print(f"RMSE (mV)               : {rmse:.4f} (Lower is better)")
    print(f"Pearson Correlation (r) : {pearson_r:.4f} (Closer to 1.0 is better)")
    print(f"Beat-Timing Error (sec) : {beat_timing_err:.4f} (Lower is better)")
    print("==================================================")
    print("=== Evaluation Pipeline Verification Completed ===")

if __name__ == "__main__":
    main()