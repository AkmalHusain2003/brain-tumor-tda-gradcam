"""
data_utils.py - Dataset Loading with Enhanced Augmentation
===========================================================
Enhanced with aggressive augmentation strategies for better generalization
"""

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split


def augment_image(img):
    """Apply enhanced aggressive augmentation to training images."""
    # PIL-compatible transforms
    pil_transform = transforms.Compose([
        # Geometric transforms
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),  # Increased from 10
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # Random shift up to 10%
            scale=(0.9, 1.1),      # Random zoom 90-110%
            shear=5                # Add slight shear
        ),
        
        # Color/intensity transforms (important for medical images)
        transforms.ColorJitter(
            brightness=0.3,  # Increased from 0.2
            contrast=0.3,    # Increased from 0.2
            saturation=0.2,  # Added saturation
            hue=0.05         # Added slight hue variation
        ),
        
        # Blur and noise
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])
    
    # Apply PIL transforms
    img_pil = Image.fromarray(img.astype(np.uint8))
    img_transformed = pil_transform(img_pil)
    
    # Convert to tensor for RandomErasing, then back to numpy
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.3,              # 30% chance
            scale=(0.02, 0.15), # Erase 2-15% of image
            ratio=(0.3, 3.3),   # Aspect ratio
            value='random'      # Random pixel values
        ),
    ])
    
    img_tensor = tensor_transform(img_transformed)
    # Convert back: (C, H, W) -> (H, W, C) and scale to [0, 255]
    img_final = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    return img_final


def augment_image_light(img):
    """Apply lighter augmentation (useful for validation-time augmentation)."""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ])
    img_pil = Image.fromarray(img.astype(np.uint8))
    return np.array(transform(img_pil))


def load_dataset(data_dir, target_size=(224, 224), test_size=0.2, augment=True, 
                 augment_multiplier=2):
    """
    Load MRI dataset with optional enhanced augmentation.
    
    Args:
        data_dir: path to dataset
        target_size: resize images to this
        test_size: fraction for test set
        augment: whether to augment training data
        augment_multiplier: how many augmented copies per image (default: 2)
    
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    data_dir = Path(data_dir)
    
    pos_dir = None
    neg_dir = None
    
    for name in ['positive', 'yes', 'tumor', '1']:
        candidate = data_dir / name
        if candidate.exists():
            pos_dir = candidate
            break
    
    for name in ['negative', 'no', 'normal', '0']:
        candidate = data_dir / name
        if candidate.exists():
            neg_dir = candidate
            break
    
    if pos_dir is None or neg_dir is None:
        raise ValueError(f"Could not find pos/neg dirs in {data_dir}")
    
    images = []
    labels = []
    
    print(f"\n{'='*70}\nLOADING DATASET\n{'='*70}")
    print(f"Positive: {pos_dir}")
    print(f"Negative: {neg_dir}")
    print(f"Augment:  {'ON' if augment else 'OFF'}")
    if augment:
        print(f"Augment Multiplier: {augment_multiplier}x (creates {augment_multiplier} versions per image)")
    
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        for path in pos_dir.glob(ext):
            try:
                img = Image.open(path).convert('RGB').resize(target_size)
                images.append(np.array(img))
                labels.append(1)
            except Exception:
                pass
    
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        for path in neg_dir.glob(ext):
            try:
                img = Image.open(path).convert('RGB').resize(target_size)
                images.append(np.array(img))
                labels.append(0)
            except Exception:
                pass
    
    if len(images) == 0:
        raise ValueError("No images found")
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Loaded: {len(images)} images ({np.mean(labels)*100:.1f}% positive)")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    val_split = test_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split, random_state=42, stratify=y_temp
    )
    
    # Apply augmentation
    if augment:
        print("\nApplying augmentation...")
        augmented_images = []
        augmented_labels = []
        
        # Create multiple augmented versions of each training image
        for i in range(augment_multiplier):
            print(f"  Creating augmentation set {i+1}/{augment_multiplier}...", end="\r")
            X_aug = np.array([augment_image(img) for img in X_train])
            augmented_images.append(X_aug)
            augmented_labels.append(y_train.copy())
        
        print(f"Created {augment_multiplier} augmented versions        ")
        
        # Combine original + all augmented versions
        X_train = np.concatenate([X_train] + augmented_images)
        y_train = np.concatenate([y_train] + augmented_labels)
        
        # Shuffle the combined dataset
        shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
    
    print(f"\n{'='*70}")
    print("FINAL SPLIT")
    print(f"{'='*70}")
    print(f"Train: {len(X_train):5d} images ({np.mean(y_train)*100:.1f}% positive)")
    print(f"Val:   {len(X_val):5d} images ({np.mean(y_val)*100:.1f}% positive)")
    print(f"Test:  {len(X_test):5d} images ({np.mean(y_test)*100:.1f}% positive)")
    if augment:
        original_train_size = len(X_temp) * (1 - val_split)
        augmentation_ratio = len(X_train) / original_train_size
        print(f"\nAugmentation: {augmentation_ratio:.1f}x increase in training data")
    print(f"{'='*70}\n")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_dataset_with_tta(data_dir, target_size=(224, 224), test_size=0.2, 
                          augment=True, augment_multiplier=2, tta_val=False):
    """
    Load dataset with optional Test-Time Augmentation (TTA) for validation.
    
    Args:
        data_dir: path to dataset
        target_size: resize images to this
        test_size: fraction for test set
        augment: whether to augment training data
        augment_multiplier: how many augmented copies per training image
        tta_val: whether to apply light augmentation to validation set
    
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    # Load dataset normally
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(
        data_dir, target_size, test_size, augment, augment_multiplier
    )
    
    # Optionally augment validation set (Test-Time Augmentation)
    if tta_val and augment:
        print("Applying Test-Time Augmentation to validation set...")
        X_val_aug = np.array([augment_image_light(img) for img in X_val])
        X_val = np.concatenate([X_val, X_val_aug])
        y_val = np.concatenate([y_val, y_val.copy()])
        print(f"Validation set augmented: {len(X_val)} images\n")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)