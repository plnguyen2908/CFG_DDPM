import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super().__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.ln_pre = nn.LayerNorm([size * size, channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
        )
        self.ln_post = nn.LayerNorm([size * size, channels])

    def forward(self, x):
        x_reshaped = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln_pre(x_reshaped)
        attention_out, _ = self.mha(x_ln, x_ln, x_ln)
        attention_out += x_reshaped
        ff_in = self.ln_post(attention_out)
        ff_out = self.ff_self(ff_in)
        ff_out += attention_out
        return ff_out.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super(DoubleConv, self).__init__()
        self.residual = residual
        mid_channels = mid_channels if mid_channels is not None else out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(1, mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(1, out_channels)

    def forward(self, x):
      x1 = F.gelu(self.gn1(self.conv1(x)))
      x2 = self.gn2(self.conv2(x1))
      if self.residual:
        return F.gelu(x2 + x)
      return x2

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t).unsqueeze(2).unsqueeze(3)
        emb = emb.expand(-1, -1, x.size(2), x.size(3))
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = DoubleConv(in_channels, in_channels, residual=True)
        self.conv2 = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat((skip_x, x), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        emb = self.emb_layer(t).unsqueeze(2).unsqueeze(3)
        emb = emb.expand(-1, -1, x.size(2), x.size(3))
        return x + emb

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cpu"):
        super(UNet, self).__init__()
        self.device = device
        self.time_dim = time_dim

        # Initial convolution block
        self.inc = DoubleConv(c_in, 64)

        # Downward path
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        # Bottleneck
        self.bot = nn.Sequential(
            DoubleConv(256, 512),
            DoubleConv(512, 512),
            DoubleConv(512, 256),
        )

        # Upward path
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)

        # Output convolution
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        # Conditional embeddings
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        sin_term = torch.sin(t * inv_freq)
        cos_term = torch.cos(t * inv_freq)
        pos_enc = torch.cat([sin_term, cos_term], dim=-1)
        return pos_enc

    def forward(self, x, t, y=None):
        # Time and class conditional encoding
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # Class embedding, if available
        if hasattr(self, 'label_emb') and y is not None:
            t += self.label_emb(y.to(self.device))

        # U-Net forward pass
        x1 = self.inc(x)
        x2 = self.sa1(self.down1(x1, t))
        x3 = self.sa2(self.down2(x2, t))
        x4 = self.sa3(self.down3(x3, t))

        x4 = self.bot(x4)

        x = self.sa4(self.up1(x4, x3, t))
        x = self.sa5(self.up2(x, x2, t))
        x = self.sa6(self.up3(x, x1, t))

        return self.outc(x)