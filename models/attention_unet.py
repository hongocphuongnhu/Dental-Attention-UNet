import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Khối Conv cơ bản: Conv -> BN -> ReLU -> Dropout -> Conv -> BN -> ReLU
    Dropout2d giúp tránh overfit với dataset nhỏ (~1200 ảnh train).
    """
    def __init__(self, in_c, out_c, dropout_p=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate theo paper gốc: Attention U-Net (Oktay et al., 2018).
    
    Args:
        F_g  : số kênh của gating signal (từ decoder, tầng dưới đi lên)
        F_l  : số kênh của skip connection (từ encoder cùng tầng)
        F_int: chiều trung gian để chiếu về cùng không gian trước khi cộng,
               thường = F_g // 2 để giảm tham số
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # Chiếu gating signal về F_int kênh
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Chiếu skip connection về F_int kênh
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Tạo attention map 1 kênh, sigmoid cho ra giá trị [0,1]
        # Broadcast: (B,1,H,W) nhân với (B,C,H,W) -> áp cùng trọng số cho mọi kênh
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1  = self.W_g(g)           # gating signal đã chiếu
        x1  = self.W_x(x)           # skip connection đã chiếu
        psi = self.relu(g1 + x1)    # cộng rồi ReLU
        psi = self.psi(psi)         # attention map [0,1]
        return x * psi              # lọc skip connection


class AttentionUNet(nn.Module):
    """
    Attention U-Net cho bài toán phân đoạn ảnh X-quang nha khoa.
    Input : (B, 1, 256, 256)  — ảnh xám 1 kênh
    Output: (B, 1, 256, 256)  — logit thô (CHƯA qua sigmoid)
    
    LƯU Ý: Output là logit, KHÔNG phải xác suất.
    - Khi train : dùng BCEWithLogitsLoss (tự áp sigmoid bên trong)
    - Khi predict: cần torch.sigmoid(output) > 0.5 để ra mask nhị phân
    """
    def __init__(self, n_classes=1):
        super().__init__()

        # --- Encoder ---
        self.e1   = ConvBlock(1,   64)
        self.e2   = ConvBlock(64,  128)
        self.e3   = ConvBlock(128, 256)
        self.e4   = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2, 2)

        # --- Bottleneck ---
        self.b = ConvBlock(512, 1024)

        # --- Decoder ---
        # Tầng 4: 1024 -> 512
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.ag4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.d4  = ConvBlock(1024, 512)   # concat: 512(up) + 512(ag) = 1024

        # Tầng 3: 512 -> 256
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.ag3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.d3  = ConvBlock(512, 256)    # concat: 256 + 256 = 512

        # Tầng 2: 256 -> 128
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.ag2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.d2  = ConvBlock(256, 128)    # concat: 128 + 128 = 256

        # Tầng 1: 128 -> 64
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.ag1 = AttentionGate(F_g=64,  F_l=64,  F_int=32)
        self.d1  = ConvBlock(128, 64)     # concat: 64 + 64 = 128

        # Output: 1x1 conv ra n_classes kênh — KHÔNG có sigmoid ở đây
        self.output = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))

        # --- Bottleneck ---
        b = self.b(self.pool(s4))

        # --- Decoder (đặt tên rõ ràng để tránh nhầm lẫn) ---
        # Tầng 4
        up4  = self.up4(b)
        att4 = self.ag4(g=up4, x=s4)
        d4   = self.d4(torch.cat([up4, att4], dim=1))

        # Tầng 3
        up3  = self.up3(d4)
        att3 = self.ag3(g=up3, x=s3)
        d3   = self.d3(torch.cat([up3, att3], dim=1))

        # Tầng 2
        up2  = self.up2(d3)
        att2 = self.ag2(g=up2, x=s2)
        d2   = self.d2(torch.cat([up2, att2], dim=1))

        # Tầng 1
        up1  = self.up1(d2)
        att1 = self.ag1(g=up1, x=s1)
        d1   = self.d1(torch.cat([up1, att1], dim=1))

        # Trả về LOGIT thô — BCEWithLogitsLoss sẽ xử lý sigmoid khi train
        return self.output(d1)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_input = torch.randn((1, 1, 256, 256))
    model      = AttentionUNet()
    output     = model(test_input)

    print("--- KIỂM TRA KIẾN TRÚC MÔ HÌNH ---")
    print(f"Input  shape : {test_input.shape}")
    print(f"Output shape : {output.shape}  (logit, chưa qua sigmoid)")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng tham số : {total_params:,}")

    assert output.shape == (1, 1, 256, 256), "Lỗi shape!"
    print("Kiến trúc khớp hoàn toàn!")