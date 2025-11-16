"""
Data Exploration Utilities
Analyze and visualize the Tiny ImageNet dataset
"""

import os
import sys
import random
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_class_names(class_names_file="data/class_mappings/class_names.txt"):
    """Load human-readable class names"""
    with open(class_names_file, 'r') as f:
        class_names = [line.strip() for line in f]
    return class_names


def analyze_dataset_structure(data_dir="data/raw/tiny-imagenet-200"):
    """Analyze the structure of the dataset"""
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"

    # Count training samples per class
    train_counts = {}
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            images_dir = class_dir / "images"
            if images_dir.exists():
                num_images = len(list(images_dir.glob("*.JPEG")))
                train_counts[class_dir.name] = num_images

    # Count validation samples
    val_images = list((val_dir / "images").glob("*.JPEG")) if (val_dir / "images").exists() else []

    print("="*80)
    print("DATASET STRUCTURE ANALYSIS")
    print("="*80)
    print(f"\nTraining Set:")
    print(f"  Number of classes:      {len(train_counts)}")
    print(f"  Total images:           {sum(train_counts.values()):,}")
    print(f"  Images per class:       {list(train_counts.values())[0] if train_counts else 0}")

    print(f"\nValidation Set:")
    print(f"  Total images:           {len(val_images):,}")

    print("\n" + "="*80 + "\n")

    return train_counts


def visualize_class_distribution(train_counts, save_path="plots/exploration"):
    """Visualize the distribution of samples across classes"""
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Class distribution histogram
    plt.figure(figsize=(12, 6))
    counts = list(train_counts.values())
    plt.hist(counts, bins=20, edgecolor='black', color='skyblue')
    plt.xlabel('Number of Images', fontsize=12)
    plt.ylabel('Number of Classes', fontsize=12)
    plt.title('Distribution of Images per Class', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Class distribution plot saved to {save_dir}/class_distribution.png")


def visualize_sample_images(
    data_dir="data/raw/tiny-imagenet-200",
    class_names_file="data/class_mappings/class_names.txt",
    num_classes=10,
    num_samples_per_class=5,
    save_path="plots/exploration"
):
    """Visualize sample images from random classes"""
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(data_dir)
    train_dir = data_path / "train"

    # Load class names
    class_names = load_class_names(class_names_file)

    # Get random classes
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    random_classes = random.sample(class_dirs, min(num_classes, len(class_dirs)))

    # Create figure
    fig, axes = plt.subplots(num_classes, num_samples_per_class, figsize=(15, 3*num_classes))

    for i, class_dir in enumerate(random_classes):
        images_dir = class_dir / "images"
        image_files = list(images_dir.glob("*.JPEG"))
        random_images = random.sample(image_files, min(num_samples_per_class, len(image_files)))

        # Get class name
        class_id = class_dir.name
        try:
            class_idx = int(class_id[1:])  # Remove 'n' prefix
            class_name = class_names[class_idx] if class_idx < len(class_names) else class_id
        except:
            class_name = class_id

        for j, img_file in enumerate(random_images):
            img = Image.open(img_file).convert('RGB')

            if num_classes == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]

            ax.imshow(img)
            ax.axis('off')

            if j == 0:
                ax.set_title(f'{class_name}', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_dir / 'sample_images.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Sample images saved to {save_dir}/sample_images.png")


def analyze_image_statistics(data_dir="data/raw/tiny-imagenet-200", num_samples=1000):
    """Analyze image statistics (mean, std, size distribution)"""
    data_path = Path(data_dir)
    train_dir = data_path / "train"

    # Collect sample images
    all_images = []
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            images_dir = class_dir / "images"
            all_images.extend(list(images_dir.glob("*.JPEG")))

    # Random sample
    sample_images = random.sample(all_images, min(num_samples, len(all_images)))

    # Calculate statistics
    means = []
    stds = []
    sizes = []

    transform = transforms.ToTensor()

    print(f"Analyzing {len(sample_images)} sample images...")

    for img_path in sample_images:
        try:
            img = Image.open(img_path).convert('RGB')
            sizes.append(img.size)

            img_tensor = transform(img)
            means.append(img_tensor.mean(dim=[1, 2]).numpy())
            stds.append(img_tensor.std(dim=[1, 2]).numpy())
        except Exception as e:
            continue

    # Calculate overall statistics
    mean_rgb = np.mean(means, axis=0)
    std_rgb = np.mean(stds, axis=0)

    print("\n" + "="*80)
    print("IMAGE STATISTICS")
    print("="*80)
    print(f"\nChannel-wise Mean (RGB):  [{mean_rgb[0]:.4f}, {mean_rgb[1]:.4f}, {mean_rgb[2]:.4f}]")
    print(f"Channel-wise Std (RGB):   [{std_rgb[0]:.4f}, {std_rgb[1]:.4f}, {std_rgb[2]:.4f}]")

    # Size distribution
    size_counter = Counter(sizes)
    print(f"\nImage Sizes:")
    for size, count in size_counter.most_common(5):
        print(f"  {size}: {count} images ({count/len(sizes)*100:.1f}%)")

    print("="*80 + "\n")

    return {
        'mean': mean_rgb.tolist(),
        'std': std_rgb.tolist(),
        'sizes': {str(k): v for k, v in size_counter.items()}
    }


def visualize_augmentations(
    data_dir="data/raw/tiny-imagenet-200",
    save_path="plots/exploration"
):
    """Visualize different augmentation techniques"""
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get a sample image
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    sample_class = random.choice(class_dirs)
    images_dir = sample_class / "images"
    sample_image_path = random.choice(list(images_dir.glob("*.JPEG")))

    # Load image
    original_img = Image.open(sample_image_path).convert('RGB')

    # Define augmentations
    augmentations = {
        'Original': transforms.Compose([transforms.Resize((64, 64))]),
        'Random Crop': transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.8, 1.0))
        ]),
        'Horizontal Flip': transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=1.0)
        ]),
        'Color Jitter': transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
        ]),
        'Rotation': transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomRotation(15)
        ]),
        'Grayscale': transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomGrayscale(p=1.0)
        ])
    }

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (aug_name, aug_transform) in enumerate(augmentations.items()):
        aug_img = aug_transform(original_img)
        axes[idx].imshow(aug_img)
        axes[idx].set_title(aug_name, fontsize=12, fontweight='bold')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_dir / 'augmentation_examples.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Augmentation examples saved to {save_dir}/augmentation_examples.png")


def create_exploration_report(save_path="results/data_exploration_report.json"):
    """Create comprehensive data exploration report"""
    print("\n" + "="*80)
    print("CREATING DATA EXPLORATION REPORT")
    print("="*80 + "\n")

    # Analyze structure
    train_counts = analyze_dataset_structure()

    # Visualize distribution
    visualize_class_distribution(train_counts)

    # Sample images
    visualize_sample_images()

    # Image statistics
    stats = analyze_image_statistics()

    # Augmentations
    visualize_augmentations()

    # Save report
    report = {
        'num_classes': len(train_counts),
        'total_train_images': sum(train_counts.values()),
        'images_per_class': list(train_counts.values())[0] if train_counts else 0,
        'image_statistics': stats
    }

    with open(save_path, 'w') as f:
        json.dump(report, f, indent=4)

    print(f"\n✅ Exploration report saved to {save_path}")
    print("="*80 + "\n")


def main():
    """Main exploration function"""
    import argparse

    parser = argparse.ArgumentParser(description="Data Exploration Utility")
    parser.add_argument(
        "--action",
        type=str,
        default="all",
        choices=["structure", "distribution", "samples", "statistics", "augmentations", "all"],
        help="What to explore"
    )

    args = parser.parse_args()

    if args.action == "structure":
        analyze_dataset_structure()
    elif args.action == "distribution":
        train_counts = analyze_dataset_structure()
        visualize_class_distribution(train_counts)
    elif args.action == "samples":
        visualize_sample_images()
    elif args.action == "statistics":
        analyze_image_statistics()
    elif args.action == "augmentations":
        visualize_augmentations()
    elif args.action == "all":
        create_exploration_report()


if __name__ == "__main__":
    main()
