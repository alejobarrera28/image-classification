import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    Differences with respect to riginal AlexNet architecture:
    - Reduced kernel_size and stride in the first Conv2d layers
    - Reduced with of fc layers
    - Added BatchNorm2d
    """

    def __init__(self, num_classes=200):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256 * 7 * 7, 1024), nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(1024, 512), nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(nn.Linear(512, num_classes))

    def forward(self, x):
        # Features
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Classifier
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    from utils.utils import count_parameters

    model = AlexNet(num_classes=200)
    count_parameters(model)
