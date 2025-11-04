import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """
    Differences with respect to riginal ResNet18 architecture:
    - Reduced kernel_size, stride and padding of first Conv2d layer
    - Removed MaxPool2d
    - Added dropout in final classifier
    """

    def __init__(self, num_classes=200):
        super(ResNet18, self).__init__()

        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Layer 1 - 64 channels, 2 blocks
        self.layer1_block1 = BasicBlock(64, 64, stride=1)
        self.layer1_block2 = BasicBlock(64, 64, stride=1)

        # Layer 2 - 128 channels, 2 blocks
        downsample2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(128)
        )
        self.layer2_block1 = BasicBlock(64, 128, stride=2, downsample=downsample2)
        self.layer2_block2 = BasicBlock(128, 128, stride=1)

        # Layer 3 - 256 channels, 2 blocks
        downsample3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
        )
        self.layer3_block1 = BasicBlock(128, 256, stride=2, downsample=downsample3)
        self.layer3_block2 = BasicBlock(256, 256, stride=1)

        # Layer 4 - 512 channels, 2 blocks
        downsample4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
        )
        self.layer4_block1 = BasicBlock(256, 512, stride=2, downsample=downsample4)
        self.layer4_block2 = BasicBlock(512, 512, stride=1)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, num_classes))

    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Layer 1
        x = self.layer1_block1(x)
        x = self.layer1_block2(x)

        # Layer 2
        x = self.layer2_block1(x)
        x = self.layer2_block2(x)

        # Layer 3
        x = self.layer3_block1(x)
        x = self.layer3_block2(x)

        # Layer 4
        x = self.layer4_block1(x)
        x = self.layer4_block2(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    from utils.utils import count_parameters

    model = ResNet18(num_classes=200)
    count_parameters(model)
