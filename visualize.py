"""
visualize.py - Results Visualization
====================================
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_training_curves(history_s1, history_s2, save_dir):
    """Plot training curves."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    if history_s1 and 'accuracy' in history_s1:
        axes[0, 0].plot(history_s1['accuracy'], 'b-', label='Train')
        axes[0, 0].plot(history_s1['val_accuracy'], 'r--', label='Val')
        axes[0, 0].set_title('Stage 1: Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(history_s1['loss'], 'b-', label='Train')
        axes[0, 1].plot(history_s1['val_loss'], 'r--', label='Val')
        axes[0, 1].set_title('Stage 1: Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    if history_s2 and 'accuracy' in history_s2:
        axes[1, 0].plot(history_s2['accuracy'], 'g-', label='Train')
        axes[1, 0].plot(history_s2['val_accuracy'], 'orange', linestyle='--', label='Val')
        axes[1, 0].set_title('Stage 2: Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(history_s2['loss'], 'g-', label='Train')
        axes[1, 1].plot(history_s2['val_loss'], 'orange', linestyle='--', label='Val')
        axes[1, 1].set_title('Stage 2: Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'training_curves.png', dpi=150)
    plt.close()
    print(" Saved: training_curves.png")


def plot_confusion_matrix(cm, acc, auc_score, save_dir):
    """Plot confusion matrix."""
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive']
    )
    plt.title(f'Confusion Matrix\n(Acc: {acc:.3f}, AUC: {auc_score:.3f})')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=150)
    plt.close()
    print(" Saved: confusion_matrix.png")


def plot_roc_curve(fpr, tpr, auc_score, save_dir):
    """Plot ROC curve."""
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'roc_curve.png', dpi=150)
    plt.close()
    print(" Saved: roc_curve.png")


def plot_attention_sample(img, att, save_path):
    """Plot attention sample."""
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    if len(img.shape) == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    
    img_display = img / 255.0 if img.max() > 1 else img
    
    axes[0].imshow(img_display)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    im = axes[1].imshow(att, cmap='jet')
    axes[1].set_title('Attention')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    axes[2].imshow(img_display)
    axes[2].imshow(att, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def generate_plots(trainer, test_images, test_attention, save_dir):
    """Generate all plots."""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}\nVISUALIZATION\n{'='*70}")
    
    plot_training_curves(trainer.history_stage1, trainer.history_stage2, save_dir)
    
    cm = np.array(trainer.metrics['confusion_matrix'])
    plot_confusion_matrix(cm, trainer.metrics['accuracy'], trainer.metrics['auc'], save_dir)
    
    plot_roc_curve(trainer.metrics['fpr'], trainer.metrics['tpr'], trainer.metrics['auc'], save_dir)
    
    n_samples = min(6, len(test_images))
    for i in range(n_samples):
        save_path = save_dir / f'attention_sample_{i}.png'
        plot_attention_sample(test_images[i], test_attention[i], save_path)
        print(f" Saved: attention_sample_{i}.png")
    
    print(f"{'='*70}\n")