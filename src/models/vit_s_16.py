import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self):
        super(PatchEmbedding, self).__init__()
        # Conv-stem: 2-layer convolutional stem instead of single large patch projection
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.gelu1 = nn.GELU()

        self.conv2 = nn.Conv2d(64, 384, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu1(x)
        x = self.conv2(x)

        # Reshape from (B, C, H, W) to (B, N, C) for transformer
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(384, 1536)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.0)

        self.fc2 = nn.Linear(1536, 384)
        self.dropout2 = nn.Dropout(0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # Multi-head attention with 6 heads, embed_dim=384
        self.num_heads = 6
        self.head_dim = 384 // 6  # 64

        self.qkv = nn.Linear(384, 1152)  # 3 * 384 for Q, K, V
        self.attn_dropout = nn.Dropout(0.0)
        self.projection = nn.Linear(384, 384)
        self.projection_dropout = nn.Dropout(0.0)

    def forward(self, x):
        B, N, C = x.shape
        # Generate Q, K, V and reshape for multi-head attention
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / self.head_dim**0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # Combine heads and project
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.projection(x)
        x = self.projection_dropout(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()
        # Pre-normalization for attention
        self.norm_attn = nn.LayerNorm(384, eps=1e-6)
        self.attention = MultiHeadAttention()

        # Pre-normalization for MLP
        self.norm_mlp = nn.LayerNorm(384, eps=1e-6)
        self.mlp = MLP()

    def forward(self, x):
        # Self-attention with residual
        out = self.norm_attn(x)
        out = self.attention(out)
        x = x + out

        # MLP with residual
        out = self.norm_mlp(x)
        out = self.mlp(out)
        x = x + out

        return x


class ViT_S_16(nn.Module):
    """
    Differences with respect to original ViT-S/16 architecture:
    - Replaced patch embedding with conv-stem
    """

    def __init__(self, num_classes=200):
        super(ViT_S_16, self).__init__()

        # Patch embedding layer
        self.patch_embedding = PatchEmbedding()
        self.pos_embedding = nn.Parameter(torch.zeros(1, 64, 384))
        self.patch_dropout = nn.Dropout(0.0)

        # Transformer blocks - 12 layers
        self.block_0 = EncoderBlock()
        self.block_1 = EncoderBlock()
        self.block_2 = EncoderBlock()
        self.block_3 = EncoderBlock()
        self.block_4 = EncoderBlock()
        self.block_5 = EncoderBlock()
        self.block_6 = EncoderBlock()
        self.block_7 = EncoderBlock()
        self.block_8 = EncoderBlock()
        self.block_9 = EncoderBlock()
        self.block_10 = EncoderBlock()
        self.block_11 = EncoderBlock()

        # Final layers
        self.final_norm = nn.LayerNorm(384, eps=1e-6)
        self.fc_dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(384, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embedding(x)
        x = x + self.pos_embedding
        x = self.patch_dropout(x)

        # Transformer blocks
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)
        x = self.block_8(x)
        x = self.block_9(x)
        x = self.block_10(x)
        x = self.block_11(x)

        # Classifier
        x = self.final_norm(x)
        x = x.mean(dim=1)
        x = self.fc_dropout(x)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    from utils.utils import count_parameters

    model = ViT_S_16(num_classes=200)
    count_parameters(model)
