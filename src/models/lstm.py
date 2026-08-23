import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    Classical baseline based on the original NewsLies architecture.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 100, hidden_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        # outputs: [batch_size, seq_len, hidden_dim]
        # hidden: [num_layers, batch_size, hidden_dim]
        _, (hidden, _) = self.lstm(embedded)
        
        # Take the hidden state of the top layer
        out = self.fc(hidden[-1])
        return out
