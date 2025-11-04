import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        # Branch 1: 1x1 conv
        branch1x1 = self.branch1x1(x)

        # Branch 2: 1x1 conv -> 5x5 conv
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        # Branch 3: 1x1 conv -> 3x3 conv -> 3x3 conv
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Branch 4: avg pool -> 1x1 conv
        branch_pool = nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # Concatenate all branches
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        # Branch 1: 3x3 conv stride 2
        branch3x3 = self.branch3x3(x)

        # Branch 2: 1x1 conv -> 3x3 conv -> 3x3 conv stride 2
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Branch 3: max pool stride 2
        branch_pool = nn.functional.max_pool2d(x, kernel_size=3, stride=2)

        # Concatenate all branches
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        # Branch 1: 1x1 conv
        branch1x1 = self.branch1x1(x)

        # Branch 2: 1x1 conv -> 1x7 conv -> 7x1 conv
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        # Branch 3: 1x1 conv -> 7x1 conv -> 1x7 conv -> 7x1 conv -> 1x7 conv
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Branch 4: avg pool -> 1x1 conv
        branch_pool = nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # Concatenate all branches
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        # Branch 1: 1x1 conv -> 3x3 conv stride 2
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        # Branch 2: 1x1 conv -> 1x7 conv -> 7x1 conv -> 3x3 conv stride 2
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        # Branch 3: max pool stride 2
        branch_pool = nn.functional.max_pool2d(x, kernel_size=3, stride=2)

        # Concatenate all branches
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        # Branch 1: 1x1 conv
        branch1x1 = self.branch1x1(x)

        # Branch 2: 1x1 conv -> split to 1x3 and 3x1 conv
        branch3x3 = self.branch3x3_1(x)
        branch3x3_a = self.branch3x3_2a(branch3x3)
        branch3x3_b = self.branch3x3_2b(branch3x3)
        branch3x3 = torch.cat([branch3x3_a, branch3x3_b], 1)

        # Branch 3: 1x1 conv -> 3x3 conv -> split to 1x3 and 3x1 conv
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl_a = self.branch3x3dbl_3a(branch3x3dbl)
        branch3x3dbl_b = self.branch3x3dbl_3b(branch3x3dbl)
        branch3x3dbl = torch.cat([branch3x3dbl_a, branch3x3dbl_b], 1)

        # Branch 4: avg pool -> 1x1 conv
        branch_pool = nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # Concatenate all branches
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = nn.functional.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class InceptionV3(nn.Module):
    def __init__(self, num_classes=200):
        super(InceptionV3, self).__init__()

        # Initial layers
        self.conv1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2a = BasicConv2d(32, 32, kernel_size=3)
        self.conv2b = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.conv3b = BasicConv2d(64, 80, kernel_size=1)
        self.conv4a = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(3, stride=2)

        # Inception blocks - Mixed 5
        self.mixed5b = InceptionA(192, pool_features=32)
        self.mixed5c = InceptionA(256, pool_features=64)
        self.mixed5d = InceptionA(288, pool_features=64)

        # Inception blocks - Mixed 6
        self.mixed6a = InceptionB(288)
        self.mixed6b = InceptionC(768, channels_7x7=128)
        self.mixed6c = InceptionC(768, channels_7x7=160)
        self.mixed6d = InceptionC(768, channels_7x7=160)
        self.mixed6e = InceptionC(768, channels_7x7=192)

        # Auxiliary classifier
        self.aux = InceptionAux(768, num_classes)

        # Inception blocks - Mixed 7
        self.mixed7a = InceptionD(768)
        self.mixed7b = InceptionE(1280)
        self.mixed7c = InceptionE(2048)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Initial conv layers
        x = self.conv1a(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.maxpool1(x)
        x = self.conv3b(x)
        x = self.conv4a(x)
        x = self.maxpool2(x)

        # Inception blocks - Mixed 5
        x = self.mixed5b(x)
        x = self.mixed5c(x)
        x = self.mixed5d(x)

        # Inception blocks - Mixed 6
        x = self.mixed6a(x)
        x = self.mixed6b(x)
        x = self.mixed6c(x)
        x = self.mixed6d(x)
        x = self.mixed6e(x)

        # Auxiliary classifier
        if self.training:
            aux = self.aux(x)
        else:
            aux = None

        # Inception blocks - Mixed 7
        x = self.mixed7a(x)
        x = self.mixed7b(x)
        x = self.mixed7c(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if aux is not None:
            return x, aux
        return x


if __name__ == "__main__":
    from utils.utils import count_parameters

    model = InceptionV3(num_classes=200)
    count_parameters(model)
