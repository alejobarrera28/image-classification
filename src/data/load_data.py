# src/utils/data_loader.py
import os
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from src.data.transforms import get_transforms
import numpy as np


def create_indices_mathematical(
    num_classes=200, images_per_class=500, val_split=0.2, random_state=42
):
    """Create indices mathematically without loading data"""
    np.random.seed(random_state)

    train_indices = []
    val_indices = []

    val_per_class = int(images_per_class * val_split)  # 100
    train_per_class = images_per_class - val_per_class  # 400

    for class_idx in range(num_classes):
        start_idx = class_idx * images_per_class
        class_indices = list(range(start_idx, start_idx + images_per_class))

        # Split deterministically
        np.random.shuffle(class_indices)

        train_indices.extend(class_indices[:train_per_class])
        val_indices.extend(class_indices[train_per_class:])

    return train_indices, val_indices


def get_data_loaders(
    root_dir="data/curated",
    batch_size=128,
    num_workers=2,
    val_split=0.2,
    random_state=42,
):
    """
    Get train/val/test loaders using ImageFolder + smart sampling

    Args:
        root_dir: Path to curated data
        val_split: Fraction of training data to use for validation

    Returns:
        train_loader: 80,000 images (400 per class)
        val_loader: 20,000 images (100 per class)
        test_loader: 10,000 images (50 per class)
    """

    train_transform, val_transform = get_transforms()

    # 1. Load original validation data as test set (10k images)
    test_dataset = ImageFolder(
        root=os.path.join(root_dir, "val"), transform=val_transform
    )

    # 2. Create train/val split indices mathematically (no data loading needed)
    train_indices, val_indices = create_indices_mathematical(
        num_classes=200,
        images_per_class=500,
        val_split=val_split,
        random_state=random_state,
    )

    # 3. Create separate datasets with different transforms
    train_dataset = ImageFolder(
        root=os.path.join(root_dir, "train"), transform=train_transform
    )

    val_dataset = ImageFolder(
        root=os.path.join(root_dir, "train"),  # Same data as train
        transform=val_transform,  # Different transform
    )

    # 4. Create samplers for train/val split
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # 5. Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    # Print split info
    print("Dataset Split Summary:")

    print(
        f"   Train: {len(train_indices):,} images ({len(train_indices)//200} per class)"
    )

    print(f"   Val:   {len(val_indices):,} images ({len(val_indices)//200} per class)")

    print(
        f"   Test:  {len(test_dataset):,} images ({len(test_dataset)//200} per class)"
    )
    print(f"   Classes: {len(train_dataset.classes)}")
    print(f"   Val split: {val_split*100:.1f}% of training data")

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    get_data_loaders()