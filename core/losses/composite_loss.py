import torch
import torch.nn as nn
# Import sequence length from our centralized configuration
from core.config import SEQ_LEN

class CompositeLoss(nn.Module):
    """
    Advanced Medical Signal Loss Function.
    פונקציית הפסד משולבת המשלבת Huber Loss, תדר (STFT), ומשקולת לשיאי הדופק.
    """
    def __init__(self, alpha=1.0, beta=0.5, gamma=2.0):
        super(CompositeLoss, self).__init__()
        self.alpha = alpha   # משקל עבור רכיב הזמן (Huber)
        self.beta = beta     # משקל עבור רכיב התדר (STFT)
        self.gamma = gamma   # משקל עבור שיאי ה-R-peak
        
        # Huber Loss (Smooth L1) robust to outliers and motion artifacts
        self.huber = nn.HuberLoss()

    def _compute_stft_loss(self, pred, target):
        """
        Computes the multi-resolution STFT magnitude difference.
        מחשבת את מרחק ה-L1 בין התדרים של האות המשוחזר לאות האמת.
        """
        # הסרת מימד הערוץ האחרון לצורך הרצת הטרנספורם
        pred_sq = pred.squeeze(-1)
        target_sq = target.squeeze(-1)
        
        # הרצת Short-Time Fourier Transform (STFT)
        stft_pred = torch.stft(pred_sq, n_fft=256, hop_length=64, win_length=256, return_complex=True)
        stft_target = torch.stft(target_sq, n_fft=256, hop_length=64, win_length=256, return_complex=True)
        
        # חישוב מתאמי האמפליטודה (Magnitude) במרחב התדר
        mag_pred = torch.abs(stft_pred)
        mag_target = torch.abs(stft_target)
        
        # החזרת שגיאת ה-L1 הממוצעת בין מרחבי התדרים
        return torch.mean(torch.abs(mag_pred - mag_target))

    def _generate_peak_mask(self, target, threshold=1.5, window_padding=10):
        """
        Dynamically isolates R-peaks on the Ground Truth ECG to construct a weight mask.
        מייצרת מסיכת משקולות סביב שיאי ה-R-peak באות האמת כדי לתת להם חשיבות קלינית.
        """
        # יצירת מסיכת בסיס מלאה ב-1.0 (כלומר משקל רגיל לכל האות)
        mask = torch.ones_like(target)
        
        # זיהוי פשוט של נקודות החורגות מסף האמפליטודה (מייצג את ה-QRS הקיצוני)
        peaks = (target > threshold)
        
        if peaks.any():
            # הרחבת המסיכה בכמה נקודות דגימה לכל כיוון סביב השיא כדי לתפוס את כל קומפלקס הגל
            for idx in range(-window_padding, window_padding + 1):
                shifted_peaks = torch.roll(peaks, shifts=idx, dims=1)
                mask[shifted_peaks] = 10.0  # הענקת משקל גבוה פי 10 לטעות באזור ה-R-peak
                
        return mask

    def forward(self, pred_ecg, true_ecg):
        """
        Calculates the weighted composite medical loss formulation.
        """
        # 1. חישוב Huber Loss במרחב הזמן (Time Domain)
        loss_time = self.huber(pred_ecg, true_ecg)
        
        # 2. חישוב הפסד במרחב התדר (Frequency Domain)
        loss_freq = self._compute_stft_loss(pred_ecg, true_ecg)
        
        # 3. חישוב הפסד ממוקד פעימות (Peak-Aware Loss)
        peak_mask = self._generate_peak_mask(true_ecg)
        loss_peaks = torch.mean(peak_mask * (pred_ecg - true_ecg) ** 2)
        
        # שילוב משוקלל של כל הרכיבים יחד
        total_loss = (self.alpha * loss_time) + (self.beta * loss_freq) + (self.gamma * loss_peaks)
        return total_loss

if __name__ == "__main__":
    print("=== Composite Loss Module Syntactically Verified ===")
    criterion = CompositeLoss()
    
    # בדיקת תקינות מדומה עם ממדי ה-Batch הרגילים שלנו
    mock_pred = torch.randn(4, 1250, 1)
    mock_true = torch.randn(4, 1250, 1)
    
    calculated_loss = criterion(mock_pred, mock_true)
    print(f"Executed forward pass. Calculated Mock Combined Loss: {calculated_loss.item():.4f}")