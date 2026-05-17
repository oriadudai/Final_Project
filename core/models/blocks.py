import torch
import torch.nn as nn

class DCBiLSTMBlock(nn.Module):
    """
    Densely Connected Bidirectional LSTM Block.
    מייצג שכבה היררכית בודדת ברשת הצפופה.
    """
    def __init__(self, input_size, hidden_size):
        super(DCBiLSTMBlock, self).__init__()
        
        # הגדרת שכבת ה-LSTM הדו-כיוונית
        # batch_first=True גורם למודל לצפות לקלט במבנה של (Batch, Sequence_Length, Features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x_list):
        """
        x_list: רשימה של טנזורים מכל השכבות הקודמות ברשת
        """
        # 1. ביצוע ה-Concatenation הצפוף לאורך ציר ה-Features (המימד האחרון)
        # זהו המנגנון המאפשר שימוש חוזר בתכונות מתדרים שונים
        x_concat = torch.cat(x_list, dim=-1)
        
        # 2. העברת האות המשרשר דרך שכבת ה-BiLSTM הנוכחית
        # lstm_out יהיה במימד: (batch_size, seq_len, hidden_size * 2) בגלל הדו-כיווניות
        lstm_out, _ = self.lstm(x_concat)
        
        return lstm_out