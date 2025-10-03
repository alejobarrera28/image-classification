import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_loss(model_name):
    """
    Generate and save training/validation loss plot for a given model.

    Args:
        model_name: Name of the model (e.g., 'alexnet')
    """
    # Load history data
    history_path = Path(f"results/{model_name}/history.json")

    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")

    with open(history_path, "r") as f:
        history = json.load(f)

    # Create plot
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.plot(epochs, history["train_loss"], "b-", label="Training Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)

    plt.title(
        f"{model_name.upper()} - Training and Validation Loss",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Create output directory and save plot
    output_dir = Path(f"plots/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_loss.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training loss plot for a model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["alexnet"],
        help="Model name (e.g., alexnet)",
    )

    args = parser.parse_args()
    plot_training_loss(args.model)
