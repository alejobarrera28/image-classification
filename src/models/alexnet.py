import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=200):
        super(AlexNet, self).__init__()
        # Adapted for 64x64 Tiny ImageNet (original AlexNet is for 224x224)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),  # 64x64 -> 64x64
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 64x64 -> 31x31
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),  # 31x31 -> 31x31
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 31x31 -> 15x15
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),  # 15x15 -> 15x15
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # 15x15 -> 15x15
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # 15x15 -> 15x15
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 15x15 -> 7x7
        )
        # 256 channels * 7 * 7 = 12544
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(256 * 7 * 7, 4096), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def count_parameters(model):
    """Return total and trainable parameter counts for a model and print them.

    Returns a tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params


if __name__ == "__main__":
    # Quick demo when this file is run directly
    model = AlexNet(num_classes=200)
    count_parameters(model)
