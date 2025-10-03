import torch
import torch.nn as nn

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import config
from data.load_data import get_data_loaders
from models.alexnet import AlexNet


def validate(model, val_loader, criterion, device, epoch=None):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    desc = f"Epoch {epoch} [Val]" if epoch is not None else "Validation"
    pbar = tqdm(val_loader, desc=desc)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {"loss": running_loss / (batch_idx + 1), "acc": 100.0 * correct / total}
            )

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def test_model(model, test_loader, device, model_name):
    """
    Final test evaluation (only run once at the end)
    """
    print(f"\n{'='*60}")
    print(f"Final Test Evaluation: {model_name.upper()}")
    print(f"{'='*60}\n")

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = validate(model, test_loader, criterion, device)

    print(f"\nFinal Test Results:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Acc:  {test_acc:.2f}%")
    print(f"{'='*60}\n")

    return test_loss, test_acc


def test_model_top5(model, test_loader, device, model_name):
    """
    Final test evaluation computing Top-1 and Top-5 accuracy.
    Returns: (test_loss, top1_acc, top5_acc)
    """
    print(f"\n{'='*60}")
    print(f"Final Test Evaluation (Top-5): {model_name.upper()}")
    print(f"{'='*60}\n")

    criterion = nn.CrossEntropyLoss()

    model.eval()
    running_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

            # Top-1
            _, pred = outputs.topk(1, dim=1, largest=True, sorted=True)
            pred = pred.squeeze(1)
            correct1 += pred.eq(targets).sum().item()

            # Top-5
            _, pred5 = outputs.topk(
                5, dim=1, largest=True, sorted=True
            )  # shape: (batch_size, 5)

            # compare each target to the 5 predictions
            correct5 += (pred5 == targets.unsqueeze(1)).any(dim=1).sum().item()

            total += targets.size(0)

    if total == 0:
        return None, 0.0, 0.0

    test_loss = running_loss / len(test_loader)
    top1_acc = 100.0 * correct1 / total
    top5_acc = 100.0 * correct5 / total

    print(f"\nFinal Test Results (Top-5):")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Top-1 Acc: {top1_acc:.2f}%")
    print(f"   Top-5 Acc: {top5_acc:.2f}%")
    print(f"{'='*60}\n")

    return test_loss, top1_acc, top5_acc


def main():
    parser = argparse.ArgumentParser(
        description="Run final test evaluation on saved model"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["alexnet"],
        help="Model to test",
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=["top1", "top5"],
        default="top5",
        help="Which metric to compute (default: top5)",
    )

    parser.add_argument(
        "--batch_size", type=int, default=config.batch_size, help="Batch size"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to a checkpoint file (overrides default experiments/<model>/final_model.pth)",
    )

    args = parser.parse_args()

    # Respect CLI batch size
    config.batch_size = args.batch_size

    # Load data
    print("Loading data...")
    _, _, test_loader = get_data_loaders(batch_size=config.batch_size)

    if len(test_loader) == 0:
        print("Test loader is empty. Nothing to evaluate.")
        return

    # Build model
    print(f"\nLoading {args.model.upper()} model...")
    if args.model == "alexnet":
        model = AlexNet(num_classes=200).to(config.device)

    # Determine checkpoint path
    exp_model_dir = Path("experiments") / args.model
    if args.checkpoint:
        # Treat argument as filename under experiments/<model>
        name = args.checkpoint
        if not name.endswith(".pth"):
            name = name + ".pth"
        ckpt_path = exp_model_dir / name
    else:
        # Default: prefer best_model.pth, fallback to final_model.pth
        primary = exp_model_dir / "best_model.pth"
        if primary.exists():
            ckpt_path = primary
        else:
            print(f"No checkpoint found at {primary}.")
            return

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=config.device)

    try:
        model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Failed to load state dict: {e}")
        return

    # Run test
    if args.metric == "top1":
        test_loss, test_acc = test_model(model, test_loader, config.device, args.model)
        results = {"test_loss": test_loss, "test_acc": test_acc}
    else:
        test_loss, top1_acc, top5_acc = test_model_top5(
            model, test_loader, config.device, args.model
        )
        results = {"test_loss": test_loss, "top1_acc": top1_acc, "top5_acc": top5_acc}


if __name__ == "__main__":
    main()
