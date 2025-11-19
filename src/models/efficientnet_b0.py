"""
EfficientNet introduced a systematic “compound scaling” strategy to scale depth,
width, and resolution together. Combined with neural architecture search, it
produced models that were far more efficient than previous CNNs. It became
popular for achieving strong accuracy while staying lightweight and practical.
"""

import torch
import torch.nn as nn


class StochasticDepth(nn.Module):
    """Stochastic Depth (Drop Path) for regularization."""

    def __init__(self, drop_prob=0.0):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, squeeze_channels):
        super(SqueezeExcitation, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.silu1 = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.silu1(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion_ratio,
        kernel_size,
        stride,
        drop_path_rate=0.0,
    ):
        super(MBConv, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        expanded_channels = in_channels * expansion_ratio
        squeeze_channels = max(1, in_channels // 4)

        # Expansion phase (if expansion_ratio > 1)
        if expansion_ratio > 1:
            self.expand_conv = nn.Conv2d(
                in_channels, expanded_channels, kernel_size=1, stride=1, bias=False
            )
            self.expand_bn = nn.BatchNorm2d(expanded_channels)
            self.expand_silu = nn.SiLU(inplace=True)

        self.expansion_ratio = expansion_ratio

        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        self.dw_conv = nn.Conv2d(
            expanded_channels,
            expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=expanded_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(expanded_channels)
        self.dw_silu = nn.SiLU(inplace=True)

        # Squeeze-and-Excitation
        self.se = SqueezeExcitation(expanded_channels, squeeze_channels)

        # Projection phase
        self.project_conv = nn.Conv2d(
            expanded_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.project_bn = nn.BatchNorm2d(out_channels)

        # Stochastic Depth
        self.stochastic_depth = (
            StochasticDepth(drop_path_rate) if self.use_residual else None
        )

    def forward(self, x):
        identity = x

        # Expansion
        if self.expansion_ratio > 1:
            out = self.expand_conv(x)
            out = self.expand_bn(out)
            out = self.expand_silu(out)
        else:
            out = x

        # Depthwise
        out = self.dw_conv(out)
        out = self.dw_bn(out)
        out = self.dw_silu(out)

        # Squeeze-and-Excitation
        out = self.se(out)

        # Projection
        out = self.project_conv(out)
        out = self.project_bn(out)

        # Residual connection with stochastic depth
        if self.use_residual:
            out = self.stochastic_depth(out)
            out = out + identity

        return out


class EfficientNetB0(nn.Module):
    """
    Differences with respect to original EfficientNet-B0 architecture:
    - Standard dropout rate of 0.5 in classifier
    """

    def __init__(self, num_classes=200, stochastic_depth_rate=0.2):
        super(EfficientNetB0, self).__init__()

        # Calculate drop path rates (linearly increasing)
        total_blocks = 16  # Total number of MBConv blocks
        drop_path_rates = [
            stochastic_depth_rate * i / total_blocks for i in range(total_blocks)
        ]
        block_idx = 0

        # Stage 0: Initial conv
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.silu1 = nn.SiLU(inplace=True)

        # Stage 1: MBConv1, k3x3, s1, e1, i32, o16, c1
        self.stage1_block1 = MBConv(
            in_channels=32,
            out_channels=16,
            expansion_ratio=1,
            kernel_size=3,
            stride=1,
            drop_path_rate=drop_path_rates[block_idx],
        )
        block_idx += 1

        # Stage 2: MBConv6, k3x3, s2, e6, i16, o24, c2
        self.stage2_block1 = MBConv(
            in_channels=16,
            out_channels=24,
            expansion_ratio=6,
            kernel_size=3,
            stride=2,
            drop_path_rate=drop_path_rates[block_idx],
        )
        block_idx += 1
        self.stage2_block2 = MBConv(
            in_channels=24,
            out_channels=24,
            expansion_ratio=6,
            kernel_size=3,
            stride=1,
            drop_path_rate=drop_path_rates[block_idx],
        )
        block_idx += 1

        # Stage 3: MBConv6, k5x5, s2, e6, i24, o40, c2
        self.stage3_block1 = MBConv(
            in_channels=24,
            out_channels=40,
            expansion_ratio=6,
            kernel_size=5,
            stride=2,
            drop_path_rate=drop_path_rates[block_idx],
        )
        block_idx += 1
        self.stage3_block2 = MBConv(
            in_channels=40,
            out_channels=40,
            expansion_ratio=6,
            kernel_size=5,
            stride=1,
            drop_path_rate=drop_path_rates[block_idx],
        )
        block_idx += 1

        # Stage 4: MBConv6, k3x3, s2, e6, i40, o80, c3
        self.stage4_block1 = MBConv(
            in_channels=40,
            out_channels=80,
            expansion_ratio=6,
            kernel_size=3,
            stride=2,
            drop_path_rate=drop_path_rates[block_idx],
        )
        block_idx += 1
        self.stage4_block2 = MBConv(
            in_channels=80,
            out_channels=80,
            expansion_ratio=6,
            kernel_size=3,
            stride=1,
            drop_path_rate=drop_path_rates[block_idx],
        )
        block_idx += 1
        self.stage4_block3 = MBConv(
            in_channels=80,
            out_channels=80,
            expansion_ratio=6,
            kernel_size=3,
            stride=1,
            drop_path_rate=drop_path_rates[block_idx],
        )
        block_idx += 1

        # Stage 5: MBConv6, k5x5, s1, e6, i80, o112, c3
        self.stage5_block1 = MBConv(
            in_channels=80,
            out_channels=112,
            expansion_ratio=6,
            kernel_size=5,
            stride=1,
            drop_path_rate=drop_path_rates[block_idx],
        )
        block_idx += 1
        self.stage5_block2 = MBConv(
            in_channels=112,
            out_channels=112,
            expansion_ratio=6,
            kernel_size=5,
            stride=1,
            drop_path_rate=drop_path_rates[block_idx],
        )
        block_idx += 1
        self.stage5_block3 = MBConv(
            in_channels=112,
            out_channels=112,
            expansion_ratio=6,
            kernel_size=5,
            stride=1,
            drop_path_rate=drop_path_rates[block_idx],
        )
        block_idx += 1

        # Stage 6: MBConv6, k5x5, s2, e6, i112, o192, c4
        self.stage6_block1 = MBConv(
            in_channels=112,
            out_channels=192,
            expansion_ratio=6,
            kernel_size=5,
            stride=2,
            drop_path_rate=drop_path_rates[block_idx],
        )
        block_idx += 1
        self.stage6_block2 = MBConv(
            in_channels=192,
            out_channels=192,
            expansion_ratio=6,
            kernel_size=5,
            stride=1,
            drop_path_rate=drop_path_rates[block_idx],
        )
        block_idx += 1
        self.stage6_block3 = MBConv(
            in_channels=192,
            out_channels=192,
            expansion_ratio=6,
            kernel_size=5,
            stride=1,
            drop_path_rate=drop_path_rates[block_idx],
        )
        block_idx += 1
        self.stage6_block4 = MBConv(
            in_channels=192,
            out_channels=192,
            expansion_ratio=6,
            kernel_size=5,
            stride=1,
            drop_path_rate=drop_path_rates[block_idx],
        )
        block_idx += 1

        # Stage 7: MBConv6, k3x3, s1, e6, i192, o320, c1
        self.stage7_block1 = MBConv(
            in_channels=192,
            out_channels=320,
            expansion_ratio=6,
            kernel_size=3,
            stride=1,
            drop_path_rate=drop_path_rates[block_idx],
        )

        # Final conv layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.silu2 = nn.SiLU(inplace=True)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, num_classes))

    def forward(self, x):
        # Stage 0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu1(x)

        # Stage 1
        x = self.stage1_block1(x)

        # Stage 2
        x = self.stage2_block1(x)
        x = self.stage2_block2(x)

        # Stage 3
        x = self.stage3_block1(x)
        x = self.stage3_block2(x)

        # Stage 4
        x = self.stage4_block1(x)
        x = self.stage4_block2(x)
        x = self.stage4_block3(x)

        # Stage 5
        x = self.stage5_block1(x)
        x = self.stage5_block2(x)
        x = self.stage5_block3(x)

        # Stage 6
        x = self.stage6_block1(x)
        x = self.stage6_block2(x)
        x = self.stage6_block3(x)
        x = self.stage6_block4(x)

        # Stage 7
        x = self.stage7_block1(x)

        # Final conv
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.silu2(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    from utils.utils import count_parameters

    model = EfficientNetB0(num_classes=200)
    count_parameters(model)
