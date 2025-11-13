import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, stride=1, bias=False)

        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layer = DenseLayer(in_channels + i * growth_rate, growth_rate=growth_rate)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.avgpool(x)
        return x


class DenseNet121(nn.Module):
    """
    Differences with respect to riginal DenseNet121 architecture:
    - Replaced initial 7x7 conv with 3×3 stack, no maxpool
    - Increased growth rate
    - Modified transition compression θ
    - Reduced layers per dense block
    """

    def __init__(self, num_classes=200):
        super(DenseNet121, self).__init__()

        # Initial layers (small-image adaptation, replaces 7×7 + maxpool)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(48)
        self.relu3 = nn.ReLU(inplace=True)

        # Dense Block 1 - 6 layers, growth_rate=20
        self.dense1 = DenseBlock(num_layers=6, in_channels=48, growth_rate=20)
        # Transition 1 - θ=0.5
        self.trans1 = Transition(in_channels=168, out_channels=84)

        # Dense Block 2 - 8 layers, growth_rate=24
        self.dense2 = DenseBlock(num_layers=8, in_channels=84, growth_rate=24)
        # Transition 2 - θ=0.6
        self.trans2 = Transition(in_channels=276, out_channels=166)

        # Dense Block 3 - 12 layers, growth_rate=28
        self.dense3 = DenseBlock(num_layers=12, in_channels=166, growth_rate=28)
        # Transition 3 - θ=0.7
        self.trans3 = Transition(in_channels=502, out_channels=351)

        # Dense Block 4 - 8 layers, growth_rate=28
        self.dense4 = DenseBlock(num_layers=8, in_channels=351, growth_rate=28)

        # Final layers
        self.bn_final = nn.BatchNorm2d(575)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(575, num_classes)

    def forward(self, x):
        # Initial feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Dense Block 1
        x = self.dense1(x)
        x = self.trans1(x)

        # Dense Block 2
        x = self.dense2(x)
        x = self.trans2(x)

        # Dense Block 3
        x = self.dense3(x)
        x = self.trans3(x)

        # Dense Block 4
        x = self.dense4(x)

        # Classifier
        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    from utils.utils import count_parameters

    model = DenseNet121(num_classes=200)
    count_parameters(model)
