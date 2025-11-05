import torch
import torch.nn as nn


class VGG16(nn.Module):
    """
    Differences with respect to riginal VGG16 architecture:
    - Added BatchNorm after each Conv layer
    - Replaced FC stack with Global Avg Pooling
    - Added Dropout for regularization
    """

    def __init__(self, num_classes=200):
        super(VGG16, self).__init__()

        # Block 1 - 64 channels
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2 - 128 channels
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3 - 256 channels
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4 - 512 channels
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv4_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5 - 512 channels
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv5_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global Average Pooling + final classifier
        self.gap = nn.AdaptiveAvgPool2d(1)  # GAP → (N, 512, 1, 1)
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, num_classes))

    def forward(self, x):
        # Block 1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.maxpool1(x)

        # Block 2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.maxpool2(x)

        # Block 3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.maxpool3(x)

        # Block 4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.maxpool4(x)

        # Block 5
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.maxpool5(x)

        # GAP + classifier
        x = self.gap(x)  # (N, 512, 1, 1)
        x = torch.flatten(x, 1)  # (N, 512)
        x = self.fc(x)  # (N, num_classes)

        return x


if __name__ == "__main__":
    from utils.utils import count_parameters

    model = VGG16(num_classes=200)
    count_parameters(model)
