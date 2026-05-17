import torch
import torch.nn as nn
# טעינת הבלוק הבודד שיצרנו בקובץ המקביל
from core.models.blocks import DCBiLSTMBlock

class ReHeartNet(nn.Module):
    """
    רשת ה-ReHeartNet המלאה לשחזור אות ECG מתוך אות PPG
    """
    def __init__(self, input_channels=1, hidden_size=64, num_blocks=5):
        super(ReHeartNet, self).__init__()
        
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList()
        
        current_input_size = input_channels
        
        # בנייה דינמית של 5 הבלוקים הצפופים
        # ככל שמתקדמים בשכבות, ה-input_size גדל בגלל השרשור של השכבות הקודמות
        for i in range(num_blocks):
            block = DCBiLSTMBlock(input_size=current_input_size, hidden_size=hidden_size)
            self.blocks.append(block)
            
            # בכל שלב, הקלט הבא יגדל בגודל ה-Hidden Size כפול 2 (בגלל ה-Bidirectional)
            current_input_size += hidden_size * 2
            
        # השכבה הליניארית הסופית (fc) המשמשת כראש הרגרסיה לשחזור האות
        # היא מקבלת את כל הפיצ'רים המותכים מכל הרמות (Hierarchical Feature Fusion)
        self.fc = nn.Linear(current_input_size, 1)

    def forward(self, ppg):
        """
        ppg: טנזור הקלט במימד (Batch, Sequence_Length, 1)
        """
        # מערך שישמור את פלטי הביניים של כל השכבות לצורך החיבור הצפוף
        features_list = [ppg]
        
        # הזרמת האות בלולאה דרך 5 הבלוקים
        for block in self.blocks:
            block_out = block(features_list)
            features_list.append(block_out)
            
        # התכת התכונות הסופית: שרשור של הקלט המקורי וכל פלטי ה-BiLSTM
        final_fused_features = torch.cat(features_list, dim=-1)
        
        # מעבר דרך ראש הרגרסיה הליניארי לשחזור נקודתי של האות
        # output יהיה במימד: (Batch, Sequence_Length, 1)
        reconstructed_ecg = self.fc(final_fused_features)
        
        return reconstructed_ecg

if __name__ == "__main__":
    # בדיקת תקינות מהירה (Sanity Check)
    # נדמה Batch של 4 חלונות PPG, כאשר כל חלון מכיל 1250 דגימות (10 שניות ב-125Hz)
    model = ReHeartNet()
    test_input = torch.randn(4, 1250, 1)
    test_output = model(test_input)
    
    print("--- בדיקת תקינות מודל ReHeartNet ---")
    print("מימד קלט ה-PPG המדומה:", test_input.shape)
    print("מימד פלט ה-ECG המשוחזר:", test_output.shape)
    # צפוי להדפיס בדיוק את אותו המימד: torch.Size([4, 1250, 1])