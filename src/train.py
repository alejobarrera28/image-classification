# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

import os
import time
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import psutil

import sys

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.data.load_data import get_data_loaders
from src.test import validate, test_model_top5


def train_epoch(
    model, train_loader, criterion, optimizer, device, epoch, mixup_cutmix_alpha=0.0, memory_tracker=None
):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # MixUp/CutMix augmentation
        if mixup_cutmix_alpha > 0 and np.random.rand() < 0.5:
            lam = np.random.beta(mixup_cutmix_alpha, mixup_cutmix_alpha)
            index = torch.randperm(inputs.size(0)).to(device)

            if np.random.rand() < 0.5:  # MixUp
                inputs = lam * inputs + (1 - lam) * inputs[index]
            else:  # CutMix
                _, _, H, W = inputs.size()
                cut_rat = np.sqrt(1.0 - lam)
                cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
                cx, cy = np.random.randint(W), np.random.randint(H)
                x1 = np.clip(cx - cut_w // 2, 0, W)
                y1 = np.clip(cy - cut_h // 2, 0, H)
                x2 = np.clip(cx + cut_w // 2, 0, W)
                y2 = np.clip(cy + cut_h // 2, 0, H)
                inputs[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]
                lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

            targets_b = targets[index]
        else:
            lam, targets_b = 1.0, targets

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = lam * criterion(outputs, targets) + (1 - lam) * criterion(
            outputs, targets_b
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track peak memory during training (only in first epoch)
        # Peak occurs after optimizer.step() when optimizer state is initialized
        if memory_tracker is not None:
            # Measure CPU (RSS) + GPU memory
            cpu_memory_mb = memory_tracker['process'].memory_info().rss / 1024**2
            if device.type == 'mps':
                gpu_memory_mb = torch.mps.driver_allocated_memory() / 1024**2
            elif device.type == 'cuda':
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024**2
            else:
                gpu_memory_mb = 0

            total_memory_mb = cpu_memory_mb + gpu_memory_mb
            memory_tracker['peak_mb'] = max(memory_tracker['peak_mb'], total_memory_mb)

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        pbar.set_postfix(
            {"loss": running_loss / (batch_idx + 1), "acc": 100.0 * correct / total}
        )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def train_model(model, model_name, train_loader, val_loader, config, resume_from=None):
    """
    Main training function

    Args:
        model: PyTorch model to train
        model_name: Name for saving (e.g., 'cnn', 'vgg', 'vit')
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration module
        resume_from: Path to checkpoint to resume from (optional)
    """

    # Create results directory
    exp_dir = Path("results") / model_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if config.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    # Learning rate scheduler
    # Use total_epochs for scheduler if specified, otherwise use epochs
    scheduler_total_epochs = getattr(config, "total_epochs", config.epochs)
    warmup_epochs = getattr(config, "warmup_epochs", 0)
    warmup_start_lr = getattr(config, "warmup_start_lr", 1e-6)

    if config.scheduler == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_total_epochs - warmup_epochs,
            eta_min=config.min_lr,
        )
    else:
        main_scheduler = None

    # Add warmup if specified
    if warmup_epochs > 0 and main_scheduler is not None:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=warmup_start_lr / config.learning_rate,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = main_scheduler

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "optimizer": config.optimizer,
            "scheduler": config.scheduler,
        },
    }

    # Add optional config parameters only if they exist
    if hasattr(config, "mixup_cutmix_alpha") and config.mixup_cutmix_alpha > 0:
        history["config"]["mixup_cutmix_alpha"] = config.mixup_cutmix_alpha

    best_val_acc = 0.0
    best_epoch = 0
    start_epoch = 1
    start_time = time.time()

    # Resume from checkpoint if specified
    if resume_from:
        print(f"\n{'='*60}")
        print(f"Resuming from checkpoint: {resume_from}")
        print(f"{'='*60}")
        checkpoint = torch.load(resume_from, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", checkpoint.get("val_acc", 0.0))
        best_epoch = checkpoint.get("best_epoch", checkpoint["epoch"])

        # Load scheduler state if it exists
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load history if it exists
        history_file = exp_dir / "history.json"
        if history_file.exists():
            with open(history_file, "r") as f:
                history = json.load(f)

        print(f"Resuming from epoch {start_epoch}")
        print(f"Best validation accuracy so far: {best_val_acc:.2f}%")
        print(f"{'='*60}\n")

    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Device: {config.device}")
    print(f"Epochs: {start_epoch} to {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Optimizer: {config.optimizer}")
    print(f"Scheduler: {config.scheduler}")
    if warmup_epochs > 0:
        print(f"Warmup: {warmup_epochs} epochs (start_lr={warmup_start_lr})")
    if hasattr(config, "mixup_cutmix_alpha") and config.mixup_cutmix_alpha > 0:
        print(f"MixUp/CutMix: enabled (alpha={config.mixup_cutmix_alpha})")
    print(f"{'='*60}\n")

    # Track peak training memory during first epoch
    # Memory tracker measures peak during forward/backward pass
    memory_tracker = {
        'process': psutil.Process(os.getpid()),
        'peak_mb': 0
    }
    peak_training_memory_mb = None

    # Training loop
    for epoch in range(start_epoch, config.epochs + 1):
        # Train (pass memory tracker only on first epoch)
        mixup_cutmix_alpha = getattr(config, "mixup_cutmix_alpha", 0.0)
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            config.device,
            epoch,
            mixup_cutmix_alpha,
            memory_tracker=memory_tracker if epoch == start_epoch else None
        )

        # Report peak memory after first epoch
        if epoch == start_epoch and memory_tracker['peak_mb'] > 0:
            peak_training_memory_mb = memory_tracker['peak_mb']
            print(f"  Peak training memory: {peak_training_memory_mb:.1f} MB")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, config.device, epoch)

        # Update learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Print epoch summary
        print(f"\nEpoch {epoch}/{config.epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
            }
            if scheduler is not None:
                checkpoint_dict["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(checkpoint_dict, exp_dir / "best_model.pth")
            print(f"  ✅ New best model saved! (Val Acc: {val_acc:.2f}%)")

        # Save checkpoint every N epochs
        if epoch % config.save_freq == 0:
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
            }
            if scheduler is not None:
                checkpoint_dict["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(checkpoint_dict, exp_dir / f"checkpoint_epoch_{epoch}.pth")

        print("-" * 60)

    # Training complete
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"{'='*60}\n")

    # Save final model
    torch.save(model.state_dict(), exp_dir / "final_model.pth")

    # Add total training time to history
    history["total_training_time_seconds"] = total_time
    history["total_training_time_hours"] = total_time / 3600

    # Add peak training memory to history
    if peak_training_memory_mb is not None:
        history["peak_training_memory_mb"] = peak_training_memory_mb

    # Save training history
    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=4)

    return history, best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train Tiny ImageNet Models")

    # Model to use (required)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["alexnet", "resnet", "densenet", "vgg", "vit", "efficientnet"],
        help="Model to train",
    )

    # General network params
    parser.add_argument(
        "--batch_size", type=int, default=config.batch_size, help="Batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=config.epochs, help="Number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=config.learning_rate, help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=config.weight_decay, help="Weight decay"
    )

    # Optimizer selection
    parser.add_argument(
        "--optimizer",
        type=str,
        default=config.optimizer,
        choices=["sgd", "adam", "adamw", "rmsprop"],
        help="Optimizer",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=config.scheduler,
        choices=["cosine", "none"],
        help="LR scheduler",
    )

    # Learning rate warmup
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=config.warmup_epochs,
        help="Number of warmup epochs (0=disabled)",
    )
    parser.add_argument(
        "--warmup_start_lr",
        type=float,
        default=config.warmup_start_lr,
        help="Starting LR for warmup phase",
    )

    # MixUp/CutMix augmentation
    parser.add_argument(
        "--mixup_cutmix_alpha",
        type=float,
        default=0.0,
        help="MixUp/CutMix alpha (0=disabled, 0.2-1.0=enabled)",
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g., results/vgg/best_model.pth)",
    )

    # Total epochs for scheduler (useful for batch training)
    parser.add_argument(
        "--total_epochs",
        type=int,
        default=None,
        help="Total epochs for LR scheduler (defaults to --epochs). Use when training in batches.",
    )

    args = parser.parse_args()

    # Override config with command-line arguments
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.optimizer = args.optimizer
    config.scheduler = args.scheduler
    config.mixup_cutmix_alpha = args.mixup_cutmix_alpha
    config.total_epochs = args.total_epochs if args.total_epochs else args.epochs
    config.warmup_epochs = args.warmup_epochs
    config.warmup_start_lr = args.warmup_start_lr

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=config.batch_size,
        val_split=0.2,
    )

    # Load model
    print(f"\nLoading {args.model.upper()} model...")
    if args.model == "alexnet":
        from src.models.alexnet import AlexNet

        model = AlexNet(num_classes=200).to(config.device)
    elif args.model == "resnet":
        from src.models.resnet18 import ResNet18

        model = ResNet18(num_classes=200).to(config.device)

    elif args.model == "densenet":
        from src.models.densenet121 import DenseNet121

        model = DenseNet121(num_classes=200).to(config.device)

    elif args.model == "vgg":
        from src.models.vgg16 import VGG16

        model = VGG16(num_classes=200).to(config.device)

    elif args.model == "vit":
        from src.models.vit_s_16 import ViT_S_16

        model = ViT_S_16(num_classes=200).to(config.device)

    elif args.model == "efficientnet":
        from src.models.efficientnet_b0 import EfficientNetB0

        model = EfficientNetB0(num_classes=200).to(config.device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Train the model
    history, best_val_acc = train_model(
        model, args.model, train_loader, val_loader, config, resume_from=args.resume
    )

    # Load best model for test
    checkpoint = torch.load(f"results/{args.model}/best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']}")

    # Final test evaluation
    test_loss, test_acc, test_top5_acc = test_model_top5(
        model, test_loader, config.device, args.model
    )

    # Save test results
    exp_dir = Path("results") / args.model
    test_results = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_top5_acc": test_top5_acc,
    }
    with open(exp_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=4)


if __name__ == "__main__":
    main()
