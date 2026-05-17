import os
import torch
import torch.nn as nn
import torch.optim as optim
# ייבוא הפרמטרים המרכזיים שקבענו בקובץ הקונפיגורציה
from core.config import DEVICE, BATCH_SIZE, SEQ_LEN, LEARNING_RATE, EPOCHS, CHECKPOINT_DIR
# ייבוא רשת ה-ReHeartNet שבנינו
from core.models.reheartnet import ReHeartNet

# ננסה לייבא את wandb. אם הוא לא מותקן בסביבה, נדמה אותו כדי שהקוד לא יקרוס
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs a single training epoch across all training batches.
    מריצה מחזור אימון יחיד על פני כל קבוצות הנתונים של ה-Train.
    """
    model.train()
    running_loss = 0.0
    
    for batch_idx, (ppg, ecg) in enumerate(dataloader):
        # העברת הטנזורים להתקן העיבוד הנבחר (CPU או GPU)
        ppg, ecg = ppg.to(device), ecg.to(device)
        
        # איפוס הגרדיאנטים של האופטימייזר מהצעד הקודם
        optimizer.zero_grad()
        
        # Propagation קדימה: הרצת אות ה-PPG דרך הרשת לקבלת ה-ECG המשוחזר
        predictions = model(ppg)
        
        # חישוב פונקציית ההפסד (כרגע MSE כברירת מחדל של המאמר המקורי)
        loss = criterion(predictions, ecg)
        
        # Propagation אחורה: חישוב הנגזרות והגרדיאנטים
        loss.backward()
        
        # עדכון משקולות המודל בהתאם לגרדיאנטים שחושבו
        optimizer.step()
        
        running_loss += loss.item()
        
    # החזרת ממוצע פונקציית ההפסד עבור ה-Epoch הנוכחי
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model performance on the validation split.
    מעריכה את ביצועי המודל על נתוני הבדיקה (Validation) ללא עדכון משקולות.
    """
    model.eval()
    running_loss = 0.0
    
    # חוסך זיכרון וזמן חישוב ע"י ביטול מנגנון מעקב הגרדיאנטים של PyTorch
    with torch.no_grad():
        for ppg, ecg in dataloader:
            ppg, ecg = ppg.to(device), ecg.to(device)
            predictions = model(ppg)
            loss = criterion(predictions, ecg)
            running_loss += loss.item()
            
    return running_loss / len(dataloader)


def main():
    print("=== Initializing ReHeartNet Training Pipeline ===")
    
    # 1. אתחול ה-Weights & Biases לצורך מעקב גרפים, אם הספריה מותקנת
    if WANDB_AVAILABLE:
        wandb.init(
            project="ppg2ecg-reheartnet",
            config={
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "sequence_length": SEQ_LEN,
                "device": str(DEVICE)
            }
        )
        print("Weights & Biases logger successfully initialized.")
    else:
        print("Wandb not found. Running training loop with local console logging only.")

    # 2. אתחול המודל, פונקציית ההפסד והאופטימייזר
    model = ReHeartNet().to(DEVICE)
    criterion = nn.MSELoss()  # פונקציית ההפסד המקורית מהמאמר (RMSE/MSE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. יצירת דאטא מדומה (Mock Data) לצורך בדיקת שלמות המערכת התחבירית
    # אנו מדמים 3 קבוצות (Batches) עבור אימון ו-2 קבוצות עבור וולידציה
    print("Generating simulated medical signal tensors for verification...")
    mock_train_loader = [
        (torch.randn(BATCH_SIZE, SEQ_LEN, 1), torch.randn(BATCH_SIZE, SEQ_LEN, 1))
        for _ in range(3)
    ]
    mock_val_loader = [
        (torch.randn(BATCH_SIZE, SEQ_LEN, 1), torch.randn(BATCH_SIZE, SEQ_LEN, 1))
        for _ in range(2)
    ]

    best_val_loss = float("inf")

    # 4. לולאת האימון המרכזית (מריצה 3 מחזורים בלבד בבדיקה הסינתטית הזו)
    test_epochs = 3
    print(f"Starting execution verification loop for {test_epochs} test epochs...")
    
    for epoch in range(1, test_epochs + 1):
        train_loss = train_one_epoch(model, mock_train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, mock_val_loader, criterion, DEVICE)
        
        # הדפסת התוצאות באנגלית לחלון הטרמינל
        print(f"Epoch [{epoch}/{test_epochs}] -> Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # שליחת הנתונים לשרתי Wandb בזמן אמת
        if WANDB_AVAILABLE:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
            
        # מנגנון שמירה של ה-Model Checkpoint הטוב ביותר על בסיס ה-Validation Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_reheartnet_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved new optimal model checkpoint to: {checkpoint_path}")

    if WANDB_AVAILABLE:
        wandb.finish()
        
    print("=== Training Pipeline Verification Completed Successfully ===")

if __name__ == "__main__":
    main()