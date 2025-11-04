import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(0.0)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.dropout2 = nn.Dropout(0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super(EncoderBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.dropout1 = nn.Dropout(0.0)
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLPBlock(in_dim=embed_dim, hidden_dim=mlp_dim)

    def forward(self, x):
        # Self-attention block
        norm_x = self.ln1(x)
        attn_out = self.attn(norm_x, norm_x, norm_x)[0]
        attn_out = self.dropout1(attn_out)
        x = x + attn_out

        # MLP block
        norm_x = self.ln2(x)
        mlp_out = self.mlp(norm_x)
        mlp_out = self.dropout1(mlp_out)
        x = x + mlp_out

        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, mlp_dim):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(0.0)
        self.layers = nn.Sequential(
            *[EncoderBlock(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = self.dropout(x)
        x = self.layers(x)
        x = self.ln(x)
        return x


class ViT_B_16(nn.Module):
    """
    Differences with respect to riginal VGG16 architecture:
    - Replace PatchEmbed with ConvStem
    """

    def __init__(self, num_classes=200):
        super(ViT_B_16, self).__init__()

        # Conv stem instead of patch embedding - 768 channels
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 384, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(384, 768, kernel_size=3, stride=2, padding=1),
        )

        # Transformer encoder - 12 layers, 12 heads, 768 embed_dim, 3072 mlp_dim
        self.encoder = Encoder(embed_dim=768, num_layers=12, num_heads=12, mlp_dim=3072)

        # Final layers
        self.fc = nn.Sequential(nn.Linear(768, num_classes))

    def forward(self, x):
        # Patch embedding
        x = self.conv_stem(x)
        x = x.flatten(2).transpose(1, 2)

        # Encoder
        x = self.encoder(x)

        # Classifier
        x = x.mean(dim=1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    from utils.utils import count_parameters

    model = ViT_B_16(num_classes=200)
    count_parameters(model)
