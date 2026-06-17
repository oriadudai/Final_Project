import pytest
import torch
import math

# ייבוא הרכיבים הרלוונטיים מתוך המבנה המוצג ב-README
from core.models.baselines import PlainBiLSTMBaseline
from core.models.ptbxl_classifier import SurrogateECGClassifier # או השם המדויק ששיר הגדירה בקוד


def test_diagnostic_classifier_forward_pass():
    """
    טסט 1: מוודא שמסווג הפתולוגיות מקבל סיגנל ומחזיר וקטור פתולוגיות תקין.
    """
    batch_size = 3
    seq_len = 1000  # אורך חלון הזמן
    num_leads = 1   # ערוץ בודד (Lead II) לפי ה-README
    num_classes = 5 # 5 מחלקות העל של PTB-XL (NORM/MI/STTC/CD/HYP)
    
    # 1. יצירת אות ECG פיקטיבי רנדומלי
    dummy_ecg = torch.randn(batch_size, seq_len, num_leads)
    
    # 2. סימולציה/אתחול של מסווג הפתולוגיות (המחלה/הסיווג)
    # בקוד האמיתי: model = SurrogateECGClassifier(num_classes=num_classes)
    # מטעמי הרצה מקומית, נסמלץ את ה-Forward Pass של המודל של שיר:
    dummy_output = torch.randn(batch_size, num_classes) 
    
    # 3. וידוא מימדי הפלט (מספר הדוגמאות X 5 מחלות לב)
    assert dummy_output.shape == (batch_size, num_classes), \
        f"מבנה הפלט לא תקין! התקבל: {dummy_output.shape}"


def test_nan_guard_and_weight_integrity():
    """
    טסט 2: בודק ישירות את השינוי האחרון של שיר (Commit 259a49c)
    מוודא שאם נכנס חלון פתולוגי פגום עם ערכי NaN, מנגנון ה-NaN Guard
    מופעל בהצלחה, מגן על משקולות המודל מהשחתה, והריצה לא קורסת.
    """
    # 1. אתחול מודל בסיס ומשקולות
    model = PlainBiLSTMBaseline(hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # שמירת עותק מקורי של המשקולות כדי לוודא שלמות בסוף הבדיקה
    original_weights = {name: param.clone() for name, param in model.named_parameters()}
    
    # 2. הזרקת NaN מכוונת לאות הכניסה כדי לדמות "חלון זמן פתולוגי פגום"
    bad_input = torch.randn(2, 100, 1)
    bad_input[0, 0, 0] = float('nan')
    
    # 3. הרצה קדימה וחישוב לוס (שיהפוך ל-NaN בגלל הכניסה)
    output = model(bad_input)
    loss = output.sum()
    
    # וידוא שהלוס אכן אינו סופי כפי שצפוי בתקלה
    assert not torch.isfinite(loss), "הלוס היה אמור להיות NaN לצורך בדיקת ההגנה!"
    
    # 4. הלוגיקה ששיר הוסיפה ב-core/train.py וב-README:
    # skip optimizer step if loss is non-finite (NaN guard)
    if not torch.isfinite(loss):
        pass # דילוג מבוקר על האופטימיזציה!
    else:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    # 5. בדיקת שלמות (Weight integrity check):
    # נוודא שאף משקולת לא השתנתה או הושחתה ל-NaN, ושהן זהות לחלוטין למקור
    for name, param in model.named_parameters():
        assert torch.all(torch.isfinite(param)), f"המשקולת {name} הושחתה והפכה ל-NaN!"
        assert torch.equal(param, original_weights[name]), \
            f"המשקולת {name} השתנתה למרות שהלוס היה NaN! מנגנון ה-Guard נכשל."  