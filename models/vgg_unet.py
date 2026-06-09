"""
models/vgg_unet.py — VGG-UNet cho bài toán phân đoạn tổn thương nha khoa.

Kiến trúc (4 tầng, đồng bộ với AttentionUNet và ResUNet):
    Encoder: 4 block conv (VGG-style)
        enc1: 1   → 64   (2 conv)
        enc2: 64  → 128  (2 conv)
        enc3: 128 → 256  (3 conv)
        enc4: 256 → 512  (3 conv)
    Bottleneck: 512 → 1024 (2 conv)
    Decoder: 4 bước upsampling + skip connection từ encoder tương ứng
        dec4: 1024 + 512 → 512
        dec3: 512  + 256 → 256
        dec2: 256  + 128 → 128
        dec1: 128  + 64  → 64
    Output: 1 channel logit (binary segmentation, KHÔNG có sigmoid)

Lưu ý quan trọng:
    - Model trả về LOGIT thô (chưa qua sigmoid)
    - BCEDiceLoss trong train.py tự xử lý sigmoid bên trong
    - metrics.py cũng tự áp sigmoid trước khi tính Dice/IoU
    - Input: (B, 1, H, W) — ảnh X-quang grayscale
    - Output: (B, 1, H, W) — logit map

Cách dùng:
    from models.vgg_unet import VGGUNet
    model = VGGUNet(n_classes=1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# KHỐI CƠ BẢN: 2 lớp Conv → BN → ReLU (dùng cho cả Encoder và Decoder)
# ---------------------------------------------------------------------------
def double_conv(in_ch, out_ch):
    """
    Khối conv chuẩn của VGG: Conv(3x3) → BN → ReLU → Conv(3x3) → BN → ReLU
    padding=1 để giữ nguyên kích thước H, W
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# KHỐI DECODER: Upsample → ghép skip connection → double_conv
# ---------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    """
    Một bước giải mã (decoder step):
        1. Upsample x2 bằng bilinear (mượt hơn ConvTranspose2d)
        2. Nối (concat) với skip connection từ encoder tương ứng
        3. Double conv để học lại đặc trưng
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        """
        Args:
            in_ch   : channels từ lớp decoder trước (feature map nhỏ hơn)
            skip_ch : channels từ skip connection (encoder)
            out_ch  : channels sau khi xử lý xong
        """
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = double_conv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # Xử lý trường hợp kích thước lệch nhau do padding lẻ
        # (ví dụ input 257x257 sau pool thành 128x128, upsample lên 256 ≠ 257)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=True)

        x = torch.cat([x, skip], dim=1)   # Ghép theo chiều channel
        return self.conv(x)


# ---------------------------------------------------------------------------
# VGG-UNET CHÍNH
# ---------------------------------------------------------------------------
class VGGUNet(nn.Module):
    """
    VGG-UNet 4 tầng cho phân đoạn ảnh grayscale 1 kênh.
    Đồng bộ với AttentionUNet và ResUNet (features=[64, 128, 256, 512]).

    Encoder (VGG-style, viết lại từ đầu để nhận 1 channel):
        enc1: 1   → 64   (2 conv)
        enc2: 64  → 128  (2 conv)
        enc3: 128 → 256  (3 conv — VGG-style từ block 3 trở đi)
        enc4: 256 → 512  (3 conv)

    Bottleneck:
        512 → 1024 (2 conv)

    Decoder:
        dec4: 1024 + 512 → 512
        dec3: 512  + 256 → 256
        dec2: 256  + 128 → 128
        dec1: 128  + 64  → 64

    Output head:
        Conv(1x1): 64 → n_classes (logit thô)
    """

    def __init__(self, n_classes=1):
        super().__init__()

        # ── ENCODER ────────────────────────────────────────────────────────
        # Block 1: 2 conv, giữ nguyên spatial → Pool x2 xuống
        self.enc1 = double_conv(1, 64)      # (B, 1,   H,   W) → (B, 64,  H,    W)
        self.pool1 = nn.MaxPool2d(2, 2)     # → (B, 64,  H/2,  W/2)

        # Block 2
        self.enc2 = double_conv(64, 128)    # → (B, 128, H/2,  W/2)
        self.pool2 = nn.MaxPool2d(2, 2)     # → (B, 128, H/4,  W/4)

        # Block 3 (3 conv, VGG-style)
        self.enc3 = nn.Sequential(
            double_conv(128, 256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )                                   # → (B, 256, H/4,  W/4)
        self.pool3 = nn.MaxPool2d(2, 2)     # → (B, 256, H/8,  W/8)

        # Block 4 (3 conv, VGG-style)
        self.enc4 = nn.Sequential(
            double_conv(256, 512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )                                   # → (B, 512, H/8,  W/8)
        self.pool4 = nn.MaxPool2d(2, 2)     # → (B, 512, H/16, W/16)

        # ── BOTTLENECK ─────────────────────────────────────────────────────
        self.bottleneck = double_conv(512, 1024)  # → (B, 1024, H/16, W/16)

        # ── DECODER ────────────────────────────────────────────────────────
        # Mỗi bước: upsample x2 + skip từ encoder tương ứng
        self.dec4 = DecoderBlock(in_ch=1024, skip_ch=512, out_ch=512)
        self.dec3 = DecoderBlock(in_ch=512,  skip_ch=256, out_ch=256)
        self.dec2 = DecoderBlock(in_ch=256,  skip_ch=128, out_ch=128)
        self.dec1 = DecoderBlock(in_ch=128,  skip_ch=64,  out_ch=64)

        # ── OUTPUT HEAD ────────────────────────────────────────────────────
        # Conv 1x1: thu gọn về n_classes channel, trả LOGIT (không sigmoid)
        self.output_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # ── Encoder ────────────────────────────────────────────────────────
        e1 = self.enc1(x)                   # (B, 64,  H,    W)
        e2 = self.enc2(self.pool1(e1))      # (B, 128, H/2,  W/2)
        e3 = self.enc3(self.pool2(e2))      # (B, 256, H/4,  W/4)
        e4 = self.enc4(self.pool3(e3))      # (B, 512, H/8,  W/8)

        # ── Bottleneck ─────────────────────────────────────────────────────
        b  = self.bottleneck(self.pool4(e4))  # (B, 1024, H/16, W/16)

        # ── Decoder (mỗi bước nhận skip từ encoder tương ứng) ─────────────
        d4 = self.dec4(b,  e4)              # (B, 512, H/8,  W/8)
        d3 = self.dec3(d4, e3)              # (B, 256, H/4,  W/4)
        d2 = self.dec2(d3, e2)              # (B, 128, H/2,  W/2)
        d1 = self.dec1(d2, e1)              # (B, 64,  H,    W)

        # ── Output ─────────────────────────────────────────────────────────
        return self.output_conv(d1)         # (B, n_classes, H, W) — LOGIT THÔ


# ---------------------------------------------------------------------------
# KIỂM TRA NHANH (chạy: python models/vgg_unet.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = VGGUNet(n_classes=1)

    # Đếm tham số
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng tham số     : {total:,}")
    print(f"Tham số trainable: {trainable:,}")

    # Test forward pass với batch 2 ảnh 256x256
    dummy = torch.randn(2, 1, 256, 256)
    out   = model(dummy)
    print(f"Input shape : {dummy.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 1, 256, 256), "Shape mismatch!"
    print("✓ Forward pass OK — 4 tầng đồng bộ với AttentionUNet & ResUNet")