"""
train.py - Two-Stage Training with End-to-End Gradients
========================================================
Stage 1: CNN only
Stage 2: CNN + Dynamic GradCAM++ + TDA
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, accuracy_score


class Trainer:
    """Two-stage trainer with LR scheduling and weight decay."""
    
    def __init__(self, model, device, results_dir, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.weight_decay = weight_decay
        self.history_stage1 = {}
        self.history_stage2 = {}
        self.metrics = {}
    
    def train_stage1(self, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
        """Stage 1: Train CNN only with LR scheduling and weight decay."""
        
        train_ds = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long()
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long()
        )
        
        # FIX: drop_last=True to avoid batch_size=1 which causes BatchNorm error
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # Add weight decay (L2 regularization)
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr,
            weight_decay=self.weight_decay
        )
        
        # Add LR scheduler - ReduceLROnPlateau (monitor val_loss)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',           # Monitor validation loss (minimize)
            factor=0.5,           # Reduce LR by half
            patience=7,           # Wait 7 epochs before reducing
            min_lr=1e-7          # Minimum learning rate
        )
        
        criterion = nn.BCELoss()
        
        print(f"\n{'='*70}\nSTAGE 1: CNN TRAINING\n{'='*70}")
        print(f"Weight Decay: {self.weight_decay}")
        print(f"Initial LR:   {lr}")
        print("LR Scheduler: ReduceLROnPlateau (mode='min', factor=0.5, patience=7)")
        print("Early Stop:   Based on val_loss (patience=20)")
        print(f"{'='*70}")
        
        history = {
            'loss': [], 'val_loss': [],
            'accuracy': [], 'val_accuracy': [],
            'auc': [], 'val_auc': [],
            'lr': []  # Track learning rate
        }
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_true = []
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            for imgs, labels in train_loader:
                imgs = imgs.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model.forward(imgs, stage=1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(outputs.detach().cpu().numpy().flatten())
                train_true.extend(labels.cpu().numpy().flatten())
            
            train_loss /= len(train_loader)
            train_acc = accuracy_score(train_true, np.array(train_preds) > 0.5)
            fpr, tpr, _ = roc_curve(train_true, train_preds)
            train_auc = auc(fpr, tpr)
            
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(self.device)
                    labels = labels.float().unsqueeze(1).to(self.device)
                    
                    outputs = self.model.forward(imgs, stage=1)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    val_preds.extend(outputs.cpu().numpy().flatten())
                    val_true.extend(labels.cpu().numpy().flatten())
            
            val_loss /= len(val_loader)
            val_acc = accuracy_score(val_true, np.array(val_preds) > 0.5)
            fpr, tpr, _ = roc_curve(val_true, val_preds)
            val_auc = auc(fpr, tpr)
            
            # Step the scheduler based on validation loss
            scheduler.step(val_loss)
            
            # Track overfitting gap
            loss_gap = val_loss - train_loss
            
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            history['auc'].append(train_auc)
            history['val_auc'].append(val_auc)
            history['lr'].append(current_lr)
            
            print(f"E{epoch+1:03d} Loss:{train_loss:.4f} VLoss:{val_loss:.4f} "
                  f"Acc:{train_acc:.4f} VAcc:{val_acc:.4f} AUC:{val_auc:.4f} "
                  f"LR:{current_lr:.2e}", end="")
            
            # Warn about overfitting
            if loss_gap > 0.1:
                print(f"LossGap:{loss_gap:.3f}", end="")
            
            # Early stopping based on val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.results_dir / 'stage1_best.pth')
                print(" → Best")
            else:
                patience_counter += 1
                print()
                if patience_counter >= patience:
                    print(f"Early stop at E{epoch+1}")
                    break
        
        self.history_stage1 = history
        print(" Stage 1 done\n")
    
    def train_stage2(self, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
        """Stage 2: Train fusion with dynamic GradCAM++, LR scheduling and weight decay."""
        
        train_ds = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long()
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long()
        )
        
        # FIX: drop_last=True to avoid batch_size=1 which causes BatchNorm error
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        topology_params = list(self.model.topology_branch.parameters())
        fusion_params = [p for n, p in self.model.named_parameters() if 'fusion' in n]
        
        # Add weight decay (L2 regularization)
        optimizer = optim.Adam(
            topology_params + fusion_params, 
            lr=lr,
            weight_decay=self.weight_decay
        )
        
        # Add LR scheduler - ReduceLROnPlateau (monitor val_loss)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',           # Monitor validation loss (minimize)
            factor=0.5,           # Reduce LR by half
            patience=10,          # Wait 10 epochs before reducing (more patient in stage 2)
            min_lr=1e-7          # Minimum learning rate
        )
        
        criterion = nn.BCELoss()
        
        print(f"\n{'='*70}\nSTAGE 2: FUSION TRAINING (Dynamic GradCAM++)\n{'='*70}")
        print(f"Weight Decay: {self.weight_decay}")
        print(f"Initial LR:   {lr}")
        print("LR Scheduler: ReduceLROnPlateau (mode='min', factor=0.5, patience=10)")
        print("Early Stop:   Based on val_loss (patience=30)")
        print(f"{'='*70}")
        
        history = {
            'loss': [], 'val_loss': [],
            'accuracy': [], 'val_accuracy': [],
            'auc': [], 'val_auc': [],
            'lr': []  # Track learning rate
        }
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 30
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_true = []
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            for imgs, labels in train_loader:
                imgs = imgs.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model.forward(imgs, stage=2)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(outputs.detach().cpu().numpy().flatten())
                train_true.extend(labels.cpu().numpy().flatten())
            
            train_loss /= len(train_loader)
            train_acc = accuracy_score(train_true, np.array(train_preds) > 0.5)
            fpr, tpr, _ = roc_curve(train_true, train_preds)
            train_auc = auc(fpr, tpr)
            
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(self.device)
                    labels = labels.float().unsqueeze(1).to(self.device)
                    
                    outputs = self.model.forward(imgs, stage=2)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    val_preds.extend(outputs.cpu().numpy().flatten())
                    val_true.extend(labels.cpu().numpy().flatten())
            
            val_loss /= len(val_loader)
            val_acc = accuracy_score(val_true, np.array(val_preds) > 0.5)
            fpr, tpr, _ = roc_curve(val_true, val_preds)
            val_auc = auc(fpr, tpr)
            
            # Step the scheduler based on validation loss
            scheduler.step(val_loss)
            
            # Track overfitting gap
            loss_gap = val_loss - train_loss
            
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            history['auc'].append(train_auc)
            history['val_auc'].append(val_auc)
            history['lr'].append(current_lr)
            
            print(f"E{epoch+1:03d} Loss:{train_loss:.4f} VLoss:{val_loss:.4f} "
                  f"Acc:{train_acc:.4f} VAcc:{val_acc:.4f} AUC:{val_auc:.4f} "
                  f"LR:{current_lr:.2e}", end="")
            
            # Warn about overfitting
            if loss_gap > 0.1:
                print(f"LossGap:{loss_gap:.3f}", end="")
            
            # Early stopping based on val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.results_dir / 'stage2_best.pth')
                print(" → Best")
            else:
                patience_counter += 1
                print()
                if patience_counter >= patience:
                    print(f"Early stop at E{epoch+1}")
                    break
        
        self.history_stage2 = history
        print(" Stage 2 done\n")
    
    def evaluate(self, X_test, y_test):
        """Evaluate on test set."""
        
        print(f"\n{'='*70}\nEVALUATION\n{'='*70}")
        
        X_test_tensor = torch.from_numpy(X_test).float().to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(X_test_tensor, stage=2)
        
        y_pred_proba = outputs.cpu().numpy().flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        
        true_positives = (y_pred * y_test).sum()
        predicted_positives = y_pred.sum()
        actual_positives = y_test.sum()
        
        prec = true_positives / max(predicted_positives, 1)
        rec = true_positives / max(actual_positives, 1)
        f1 = f1_score(y_test, y_pred)
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        self.metrics = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'auc': float(auc_score),
            'sensitivity': float(sens),
            'specificity': float(spec),
            'confusion_matrix': cm.tolist(),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
        print(f"Accuracy:    {acc:.4f}")
        print(f"Precision:   {prec:.4f}")
        print(f"Recall:      {rec:.4f}")
        print(f"F1:          {f1:.4f}")
        print(f"AUC:         {auc_score:.4f}")
        print(f"Sensitivity: {sens:.4f}")
        print(f"Specificity: {spec:.4f}")
        print(f"Confusion:   TN={tn} FP={fp} FN={fn} TP={tp}")
        print("="*70)
    
    def save(self):
        """Save results."""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model': 'PyTorch CNN + GradCAM++ + TDA (End-to-End)',
            'weight_decay': self.weight_decay,
            'metrics': self.metrics,
            'history_stage1': {
                'final_lr': self.history_stage1['lr'][-1] if self.history_stage1.get('lr') else None,
                'best_val_loss': min(self.history_stage1['val_loss']) if self.history_stage1.get('val_loss') else None
            },
            'history_stage2': {
                'final_lr': self.history_stage2['lr'][-1] if self.history_stage2.get('lr') else None,
                'best_val_loss': min(self.history_stage2['val_loss']) if self.history_stage2.get('val_loss') else None
            }
        }
        
        with open(self.results_dir / 'results.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        torch.save(self.model.state_dict(), self.results_dir / 'final_model.pth')
        
        print(f" Results saved to {self.results_dir}")