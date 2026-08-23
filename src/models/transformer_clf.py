import torch
import torch.nn as nn
import warnings

class TransformerClassifier(nn.Module):
    """
    HuggingFace Transformer wrapper for AraBERT and CAMeLBERT.
    """
    def __init__(self, model_name: str, num_classes: int = 3):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError:
            warnings.warn("transformers package is required for TransformerClassifier.")
            raise
            
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.encoder.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract [CLS] token representation (or last_hidden_state pooler)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        return self.fc(cls_output)
