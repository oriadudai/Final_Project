import sys
import os

# הוספת הנתיב הנוכחי לנתיבי החיפוש של פייתון
sys.path.append(os.getcwd())

try:
    from core.models.ptbxl_classifier import SurrogateECGClassifier
    import torch
    print("✅ הייבוא הצליח!")
except Exception as e:
    print(f"❌ שגיאה בייבוא: {e}")
    sys.exit(1)

try:
    model = SurrogateECGClassifier(num_classes=5, embed_dim=128)
    print("✅ המודל נבנה בהצלחה")
    
    x = torch.randn(2, 1, 1000)
    output = model(x)
    print(f"✅ יציאת המודל תקינה: {output.shape}")
except Exception as e:
    print(f"❌ שגיאה בהרצת המודל: {e}")