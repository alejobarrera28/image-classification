# src/training/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import os
import time
import argparse
from pathlib import Path
import json
from tqdm import tqdm

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import config
from data.load_data import get_data_loaders
from training.test import validate, test_model_top5


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

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


def train_model(model, model_name, train_loader, val_loader, config):
    """
    Main training function

    Args:
        model: PyTorch model to train
        model_name: Name for saving (e.g., 'cnn', 'vgg', 'vit')
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration module
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
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    # Learning rate scheduler
    if config.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.min_lr
        )
    elif config.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10, verbose=True
        )
    else:
        scheduler = None

    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Optimizer: {config.optimizer}")
    print(f"Scheduler: {config.scheduler}")
    print(f"{'='*60}\n")

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

    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    # Training loop
    for epoch in range(1, config.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, config.device, epoch)

        # Update learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            if config.scheduler == "plateau":
                scheduler.step(val_acc)
            else:
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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                exp_dir / "best_model.pth",
            )
            print(f"  ✅ New best model saved! (Val Acc: {val_acc:.2f}%)")

        # Save checkpoint every N epochs
        if epoch % config.save_freq == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                exp_dir / f"checkpoint_epoch_{epoch}.pth",
            )

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
        choices=["alexnet", "resnet"],
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

    # Optimizer selection
    parser.add_argument(
        "--optimizer",
        type=str,
        default=config.optimizer,
        choices=["sgd", "adam", "adamw"],
        help="Optimizer",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=config.scheduler,
        choices=["cosine", "plateau", "none"],
        help="LR scheduler",
    )

    args = parser.parse_args()

    # Override config with command-line arguments
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.optimizer = args.optimizer
    config.scheduler = args.scheduler

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=config.batch_size,
        val_split=0.2,
    )

    # Load model
    print(f"\nLoading {args.model.upper()} model...")
    if args.model == "alexnet":
        from models.alexnet import AlexNet

        model = AlexNet(num_classes=200).to(config.device)
    elif args.model == "resnet":
        from models.resnet18 import ResNet18

        model = ResNet18(num_classes=200).to(config.device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Train the model
    history, best_val_acc = train_model(
        model, args.model, train_loader, val_loader, config
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
