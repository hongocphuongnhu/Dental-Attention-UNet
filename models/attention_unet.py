import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        
        # Encoder
        self.e1 = conv_block(1, 64)
        self.e2 = conv_block(64, 128)
        self.e3 = conv_block(128, 256)
        self.e4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d((2, 2))
        
        # Bottleneck
        self.b = conv_block(512, 1024)
        
        # Decoder - Chỉnh sửa lại số kênh 
        # up4: 1024 -> 512
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.ag4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.d4 = conv_block(1024, 512) # 512 (up) + 512 (ag)
        
        # up3: 512 -> 256
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.ag3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.d3 = conv_block(512, 256) # 256 (up) + 256 (ag)
        
        # up2: 256 -> 128
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.ag2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.d2 = conv_block(256, 128) # 128 (up) + 128 (ag)
        
        # up1: 128 -> 64
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.ag1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.d1 = conv_block(128, 64) # 64 (up) + 64 (ag)
        
        self.output = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        s1 = self.e1(x)
        p1 = self.pool(s1)
        s2 = self.e2(p1)
        p2 = self.pool(s2)
        s3 = self.e3(p2)
        p3 = self.pool(s3)
        s4 = self.e4(p3)
        p4 = self.pool(s4)
        
        b = self.b(p4)
        
        # Decoder với logic kênh đã sửa
        d4 = self.up4(b)
        x4 = self.ag4(g=d4, x=s4)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.d4(d4)
        
        d3 = self.up3(d4)
        x3 = self.ag3(g=d3, x=s3)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.d3(d3)
        
        d2 = self.up2(d3)
        x2 = self.ag2(g=d2, x=s2)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.d2(d2)
        
        d1 = self.up1(d2)
        x1 = self.ag1(g=d1, x=s1)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.d1(d1)
        
        return torch.sigmoid(self.output(d1))

if __name__ == "__main__":
    # 1. Khởi tạo dữ liệu giả lập (Mock data) với kích thước 256x256
    # (1: Batch size, 1: Ảnh xám, 256, 256: Rộng x Cao)
    test_input = torch.randn((1, 1, 256, 256)) 
    
    # 2. Khởi tạo mô hình
    model = AttentionUNet()
    
    # 3. Chạy thử (Inference test)
    output = model(test_input)
    
    # 4. Kiểm tra kết quả
    print("--- KIỂM TRA KIẾN TRÚC MÔ HÌNH ---")
    print(f"Kích thước ảnh đầu vào: {test_input.shape}")
    print(f"Kích thước kết quả dự đoán: {output.shape}")
    
    if output.shape == (1, 1, 256, 256):
        print("Kết quả: Kiến trúc khớp hoàn toàn (100%)!")
    else:
        print("Kết quả: Có lỗi sai lệch kích thước.")