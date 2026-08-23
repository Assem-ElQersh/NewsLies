import torch
from torch.cuda.amp import autocast, GradScaler
from src.evaluation.metrics import compute_metrics

class Trainer:
    """Unified training loop for Classical and Transformer models."""
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, device, scheduler=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
        self.scaler = GradScaler()
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            
            with autocast():
                if isinstance(batch, dict) and 'input_ids' in batch:
                    # Transformer format
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    outputs = self.model(input_ids, attention_mask)
                else:
                    # Classical format
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    
                loss = self.loss_fn(outputs, labels)
                
            self.scaler.scale(loss).backward()
            
            # Gradient clipping is especially important for Transformers/RNNs
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler:
                self.scheduler.step()
                
            total_loss += loss.item()
            
        return total_loss / max(1, len(self.train_loader))
        
    def evaluate(self, loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict) and 'input_ids' in batch:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    outputs = self.model(input_ids, attention_mask)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
        metrics = compute_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / max(1, len(loader))
        return metrics
        
    def train(self, epochs, save_path="best_model.pt", patience=2):
        best_f1 = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['macro_f1']:.4f}")
            
            if val_metrics['macro_f1'] > best_f1:
                best_f1 = val_metrics['macro_f1']
                torch.save(self.model.state_dict(), save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
                    
        # Load best weights before returning
        self.model.load_state_dict(torch.load(save_path, map_location=self.device, weights_only=True))
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        return best_f1
