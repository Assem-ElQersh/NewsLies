import torch
import torch.nn as nn
import torch.nn.functional as F

class BiGRUAttention(nn.Module):
    """
    Neural Baseline candidate for compact deployment.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 300, hidden_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        # Self-attention over the sequence
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        # output: [batch_size, seq_len, hidden_dim * 2]
        output, _ = self.gru(embedded)
        
        # Calculate attention weights
        attn_weights = F.softmax(self.attention(output).squeeze(2), dim=1)
        
        # Context vector: weighted sum of sequence outputs
        # [batch_size, 1, seq_len] x [batch_size, seq_len, hidden_dim * 2] -> [batch_size, 1, hidden_dim * 2]
        context = torch.bmm(attn_weights.unsqueeze(1), output).squeeze(1)
        
        return self.fc(context)
