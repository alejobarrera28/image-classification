import torch
import torch.nn.functional as F
from PIL import Image
import argparse
import json
from pathlib import Path
import sys
import os

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.data.transforms import get_transforms
from src.models.alexnet import AlexNet
from src.models.resnet18 import ResNet18
from src.models.densenet121 import DenseNet121
from src.models.vgg16 import VGG16
from src.models.vit_s_16 import ViT_S_16
from src.models.efficientnet_b0 import EfficientNetB0


def load_class_names(class_names_file="data/class_mappings/class_names.txt"):
    """Load human-readable class names"""
    class_names = []
    with open(class_names_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                # Format: "n01443537\tgoldfish, Carassius auratus"
                parts = line.split('\t')
                class_names.append(parts[1].split(',')[0])  # Take first name only
            else:
                class_names.append(line)
    return class_names


def load_model(model_name, checkpoint_path, device, num_classes=200):
    """Load model architecture and trained weights"""
    # Load model architecture
    if model_name == "alexnet":
        model = AlexNet(num_classes=num_classes)
    elif model_name == "resnet":
        model = ResNet18(num_classes=num_classes)
    elif model_name == "densenet":
        model = DenseNet121(num_classes=num_classes)
    elif model_name == "vgg":
        model = VGG16(num_classes=num_classes)
    elif model_name == "vit":
        model = ViT_S_16(num_classes=num_classes)
    elif model_name == "efficientnet":
        model = EfficientNetB0(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def infer_image(model, image_path, transform, device, class_names, top_k=5):
    """
    Infer the class of a single image

    Args:
        model: Trained model
        image_path: Path to image file
        transform: Preprocessing transform
        device: Device to run inference on
        class_names: List of class names
        top_k: Number of top predictions to return

    Returns:
        dict containing predictions and probabilities
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probs, top_k, dim=1)
    top_probs = top_probs.squeeze(0).cpu().numpy()
    top_indices = top_indices.squeeze(0).cpu().numpy()

    # Prepare results
    results = {
        'image_path': str(image_path),
        'predictions': []
    }

    for i in range(top_k):
        class_idx = int(top_indices[i])
        prob = float(top_probs[i])
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"

        results['predictions'].append({
            'rank': i + 1,
            'class_id': class_idx,
            'class_name': class_name,
            'probability': prob,
            'confidence': f"{prob * 100:.2f}%"
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Infer class of an image using trained model"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["alexnet", "resnet", "densenet", "vgg", "vit", "efficientnet"],
        help="Model to use for inference",
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image file to classify",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (default: results/<model>/best_model.pth)",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top predictions to show (default: 5)",
    )

    parser.add_argument(
        "--class_names",
        type=str,
        default="data/class_mappings/class_names.txt",
        help="Path to class names file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file to save results (optional)",
    )

    args = parser.parse_args()

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = f"results/{args.model}/best_model.pth"

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # Check if image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return

    # Load class names
    class_names_path = Path(args.class_names)
    if not class_names_path.exists():
        print(f"Error: Class names file not found at {class_names_path}")
        return

    print(f"\n{'='*60}")
    print(f"Image Inference: {args.model.upper()}")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image: {image_path}")
    print(f"Device: {config.device}")
    print(f"{'='*60}\n")

    # Load class names
    class_names = load_class_names(args.class_names)
    print(f"Loaded {len(class_names)} class names")

    # Load model
    print(f"Loading {args.model} model...")
    model, checkpoint = load_model(args.model, checkpoint_path, config.device)
    best_val_acc = checkpoint.get('best_val_acc', checkpoint.get('val_acc', 'N/A'))
    print(f"✅ Model loaded successfully")
    print(f"Best validation accuracy: {best_val_acc}")

    # Get preprocessing transforms (use validation transforms for inference)
    _, val_transform = get_transforms()

    # Run inference
    print(f"\nRunning inference on {image_path.name}...")
    results = infer_image(
        model,
        image_path,
        val_transform,
        config.device,
        class_names,
        top_k=args.top_k
    )

    # Display results
    print(f"\n{'='*60}")
    print(f"Top-{args.top_k} Predictions:")
    print(f"{'='*60}")
    for pred in results['predictions']:
        print(f"{pred['rank']}. {pred['class_name']}")
        print(f"   Confidence: {pred['confidence']}")
        print(f"   Class ID: {pred['class_id']}")
        print()

    # Save results to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"✅ Results saved to {output_path}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
