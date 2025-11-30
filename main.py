"""
main.py - MRI Classification Pipeline with K-Fold Cross-Validation
===================================================================
Two-stage training with dynamic GradCAM++ and TDA.
Enhanced with: K-Fold CV + Augmentation Multiplier Sweep
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold

from model import MRIClassifier
from train import Trainer
from data_utils import load_dataset
from visualize import generate_plots


def train_single_fold(fold_idx, X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
                       args, device, results_dir):
    """Train a single fold and return validation metrics."""
    
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx + 1}")
    print(f"{'='*70}")
    
    # Normalize
    X_train_norm = X_train_fold.astype(np.float32) / 255.0
    X_val_norm = X_val_fold.astype(np.float32) / 255.0
    
    # Transpose to (N, C, H, W)
    X_train_norm = X_train_norm.transpose(0, 3, 1, 2)
    X_val_norm = X_val_norm.transpose(0, 3, 1, 2)
    
    # Initialize model and trainer
    model = MRIClassifier(freeze_layers=args.freeze_layers)
    trainer = Trainer(model, device, results_dir, weight_decay=args.weight_decay)
    
    # Stage 1
    trainer.train_stage1(
        X_train_norm, y_train_fold, X_val_norm, y_val_fold,
        args.epochs_s1, args.batch_size, args.lr
    )
    
    model.load_state_dict(torch.load(results_dir / 'stage1_best.pth'))
    
    # Stage 2
    trainer.train_stage2(
        X_train_norm, y_train_fold, X_val_norm, y_val_fold,
        args.epochs_s2, args.batch_size, args.lr
    )
    
    model.load_state_dict(torch.load(results_dir / 'stage2_best.pth'))
    
    # Evaluate on validation fold
    X_val_tensor = torch.from_numpy(X_val_norm).float().to(device)
    model.eval()
    with torch.no_grad():
        outputs = model.forward(X_val_tensor, stage=2)
    
    y_pred_proba = outputs.cpu().numpy().flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_val_fold, y_pred)
    f1 = f1_score(y_val_fold, y_pred)
    auc = roc_auc_score(y_val_fold, y_pred_proba)
    
    # Calculate validation loss
    val_loss = trainer.history_stage2['val_loss'][-1] if trainer.history_stage2.get('val_loss') else float('inf')
    
    metrics = {
        'accuracy': float(accuracy),
        'f1': float(f1),
        'auc': float(auc),
        'val_loss': float(val_loss)
    }
    
    print(f"\nFold {fold_idx + 1} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC:      {auc:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    
    # Return metrics, model state, AND training history
    return metrics, model.state_dict(), trainer.history_stage1, trainer.history_stage2


def run_kfold_for_multiplier(augment_multiplier, X_all, y_all, X_test, y_test, 
                              args, device, base_results_dir, k_folds=10):
    """Run K-Fold CV for a specific augmentation multiplier."""
    
    print("\n" + "="*70)
    print(f"AUGMENTATION MULTIPLIER: {augment_multiplier}x")
    print("="*70)
    
    # Create results directory for this multiplier
    multiplier_dir = base_results_dir / f"multiplier_{augment_multiplier}"
    multiplier_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup K-Fold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = []
    fold_models = []
    
    # Run K-Fold CV
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        X_train_fold = X_all[train_idx]
        y_train_fold = y_all[train_idx]
        X_val_fold = X_all[val_idx]
        y_val_fold = y_all[val_idx]
        
        # Apply augmentation to training fold
        if augment_multiplier > 0:
            from data_utils import augment_image
            
            print(f"\n  Applying {augment_multiplier}x augmentation to fold {fold_idx + 1}...")
            augmented_images = []
            augmented_labels = []
            
            for i in range(augment_multiplier):
                X_aug = np.array([augment_image(img) for img in X_train_fold])
                augmented_images.append(X_aug)
                augmented_labels.append(y_train_fold.copy())
            
            X_train_fold = np.concatenate([X_train_fold] + augmented_images)
            y_train_fold = np.concatenate([y_train_fold] + augmented_labels)
            
            # Shuffle
            shuffle_idx = np.random.permutation(len(X_train_fold))
            X_train_fold = X_train_fold[shuffle_idx]
            y_train_fold = y_train_fold[shuffle_idx]
            
            print(f"  ✓ Training set size: {len(X_train_fold)} (original: {len(train_idx)})")
        
        # Create fold-specific directory
        fold_dir = multiplier_dir / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(exist_ok=True)
        
        # Train this fold
        metrics, model_state, history_s1, history_s2 = train_single_fold(
            fold_idx, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
            args, device, fold_dir
        )
        
        fold_results.append(metrics)
        fold_models.append({
            'fold': fold_idx + 1,
            'metrics': metrics,
            'state_dict': model_state,
            'history_stage1': history_s1,
            'history_stage2': history_s2,
            'fold_dir': fold_dir
        })
    
    # Calculate mean metrics across folds
    mean_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'f1': np.mean([r['f1'] for r in fold_results]),
        'auc': np.mean([r['auc'] for r in fold_results]),
        'val_loss': np.mean([r['val_loss'] for r in fold_results])
    }
    
    std_metrics = {
        'accuracy': np.std([r['accuracy'] for r in fold_results]),
        'f1': np.std([r['f1'] for r in fold_results]),
        'auc': np.std([r['auc'] for r in fold_results]),
        'val_loss': np.std([r['val_loss'] for r in fold_results])
    }
    
    print(f"\n{'='*70}")
    print(f"K-FOLD CV SUMMARY (Multiplier {augment_multiplier}x)")
    print(f"{'='*70}")
    print(f"Mean Accuracy: {mean_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    print(f"Mean F1:       {mean_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    print(f"Mean AUC:      {mean_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}")
    print(f"Mean Val Loss: {mean_metrics['val_loss']:.4f} ± {std_metrics['val_loss']:.4f}")
    
    # Select best model based on validation loss (lowest is best)
    best_fold_idx = np.argmin([r['val_loss'] for r in fold_results])
    best_fold = fold_models[best_fold_idx]
    
    print(f"\nBest Fold: {best_fold['fold']} (Val Loss: {best_fold['metrics']['val_loss']:.4f})")
    
    # Save best model
    best_model_path = multiplier_dir / 'best_model.pth'
    torch.save(best_fold['state_dict'], best_model_path)
    print(f" Saved best model to: {best_model_path}")
    
    # Evaluate best model on test set
    print(f"\n{'='*70}")
    print("EVALUATING BEST MODEL ON TEST SET")
    print(f"{'='*70}")
    
    model = MRIClassifier(freeze_layers=args.freeze_layers)
    model.load_state_dict(best_fold['state_dict'])
    trainer = Trainer(model, device, multiplier_dir, weight_decay=args.weight_decay)
    
    # Restore the training history from best fold
    trainer.history_stage1 = best_fold['history_stage1']
    trainer.history_stage2 = best_fold['history_stage2']
    
    # Normalize and transpose test data
    X_test_norm = X_test.astype(np.float32) / 255.0
    X_test_norm = X_test_norm.transpose(0, 3, 1, 2)
    
    trainer.evaluate(X_test_norm, y_test)
    
    # Save comprehensive results
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'augment_multiplier': augment_multiplier,
        'k_folds': k_folds,
        'best_fold': best_fold['fold'],
        'cv_mean_metrics': mean_metrics,
        'cv_std_metrics': std_metrics,
        'fold_results': fold_results,
        'test_metrics': trainer.metrics,
        'config': {
            'freeze_layers': args.freeze_layers,
            'epochs_s1': args.epochs_s1,
            'epochs_s2': args.epochs_s2,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay
        }
    }
    
    with open(multiplier_dir / 'cv_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n Results saved to: {multiplier_dir / 'cv_results.json'}")
    
    # Generate visualizations for best model
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    att_test = model.extract_attention_numpy(X_test_norm, device)
    generate_plots(trainer, X_test_norm, att_test, multiplier_dir)
    
    return results_summary


def main(args):
    """Execute K-Fold CV pipeline for multiple augmentation multipliers."""
    
    print("\n" + "="*70)
    print("MRI CLASSIFICATION: K-FOLD CV + AUGMENTATION SWEEP")
    print("="*70)
    print("Strategy: 10-Fold CV for each augmentation multiplier")
    print("Multipliers: [0, 1, 2, 3]")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    base_results_dir = Path(args.results_dir)
    base_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load full dataset WITHOUT augmentation (we'll apply it per fold)
    print(f"\n{'='*70}")
    print("LOADING BASE DATASET (No Augmentation)")
    print(f"{'='*70}")
    
    (X_train_val, y_train_val), _, (X_test, y_test) = load_dataset(
        args.data_dir,
        augment=False,  # Don't augment here
        augment_multiplier=0
    )
    
    # Augmentation multipliers to test
    augment_multipliers = [0, 1, 2, 3]
    
    all_results = {}
    
    # Loop over augmentation multipliers
    for multiplier in augment_multipliers:
        results = run_kfold_for_multiplier(
            multiplier, X_train_val, y_train_val, X_test, y_test,
            args, device, base_results_dir, k_folds=10
        )
        all_results[f'multiplier_{multiplier}'] = results
    
    # Save consolidated results
    print(f"\n{'='*70}")
    print("CONSOLIDATING RESULTS")
    print(f"{'='*70}")
    
    consolidated = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'augment_multipliers': augment_multipliers,
            'k_folds': 10,
            'freeze_layers': args.freeze_layers,
            'epochs_s1': args.epochs_s1,
            'epochs_s2': args.epochs_s2,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay
        },
        'results_by_multiplier': all_results
    }
    
    with open(base_results_dir / 'consolidated_results.json', 'w') as f:
        json.dump(consolidated, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Mult':<6} {'CV Acc':<12} {'CV F1':<12} {'Test Acc':<12} {'Test AUC':<12}")
    print("-" * 70)
    
    for mult in augment_multipliers:
        result = all_results[f'multiplier_{mult}']
        cv_acc = result['cv_mean_metrics']['accuracy']
        cv_f1 = result['cv_mean_metrics']['f1']
        test_acc = result['test_metrics']['accuracy']
        test_auc = result['test_metrics']['auc']
        
        print(f"{mult}x     {cv_acc:.4f}       {cv_f1:.4f}       {test_acc:.4f}       {test_auc:.4f}")
    
    print(f"{'='*70}")
    print(f"\n All results saved to: {base_results_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MRI Classification Pipeline with K-Fold CV')
    parser.add_argument('--data_dir', type=str, default='./brain_tumor_data')
    parser.add_argument('--results_dir', type=str, default='./results_kfold')
    parser.add_argument('--freeze_layers', type=int, default=15)
    parser.add_argument('--epochs_s1', type=int, default=50)
    parser.add_argument('--epochs_s2', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    if not Path(args.data_dir).exists():
        for alt in ['./data', './mri_data', './dataset', './brain_tumor_dataset']:
            if Path(alt).exists():
                args.data_dir = alt
                break
    
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    print(f"Data:          {args.data_dir}")
    print(f"Results:       {args.results_dir}")
    print("K-Folds:       10")
    print("Multipliers:   [0, 1, 2, 3]")
    print(f"Frozen layers: {args.freeze_layers}")
    print(f"Epochs S1:     {args.epochs_s1}")
    print(f"Epochs S2:     {args.epochs_s2}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay:  {args.weight_decay}")
    print("="*70 + "\n")
    
    main(args)