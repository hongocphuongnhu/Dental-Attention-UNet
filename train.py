import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# --- Import các file của nhóm Nhu ---
from models.attention_unet import AttentionUNet
from scripts.dataset import DentalDataset  # Đảm bảo đường dẫn này đúng
from scripts.metrics import get_metrics    # Nhu đổi thành utils.metrics nếu để trong thư mục utils

# --- Định nghĩa Combo Loss (BCE + Dice) ---
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss() # Dùng BCELoss vì mô hình Nhu đã có Sigmoid ở cuối

    def forward(self, inputs, targets, smooth=1e-6):
        # 1. Tính BCE Loss
        bce_loss = self.bce(inputs, targets)
        
        # 2. Tính Dice Loss
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth))
        
        # 3. Kết hợp (Tỉ lệ 1:1)
        return bce_loss + dice_loss

# --- Hàm Huấn Luyện Chính ---
def train_model():
    # Cấu hình "Luật chơi"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 50
    BATCH_SIZE = 8  # Nếu i9 gào rú hoặc báo Out of Memory, Nhu hạ xuống 4 hoặc 2 nhé!
    LR = 1e-4

    print(f"🚀 Bắt đầu Train hệ thống trên: {device.type.upper()}")

    # 1. Khởi tạo Băng chuyền dữ liệu (Sử dụng ảnh đã xử lý)
    data_dir = "data/processed"
    train_ds = DentalDataset(root_dir=data_dir, split='train')
    val_ds = DentalDataset(root_dir=data_dir, split='val')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"📦 Đã nạp {len(train_ds)} ảnh Train và {len(val_ds)} ảnh Validation.")

    # 2. Khởi tạo Mô hình, Loss và Optimizer
    model = AttentionUNet(n_classes=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_dice = 0.0
    os.makedirs("models_saved", exist_ok=True) # Tạo thư mục lưu kết quả

    # 3. Bắt đầu Vòng lặp Huấn luyện
    for epoch in range(EPOCHS):
        # -- HUẤN LUYỆN (TRAIN) --
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            # Quá trình học
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        # -- KIỂM TRA BÀI CŨ (VALIDATION) --
        model.eval()
        val_dice_total = 0
        val_iou_total = 0
        
        with torch.no_grad(): # Không cập nhật gradient lúc kiểm tra
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                dice, iou = get_metrics(outputs, masks)
                val_dice_total += dice
                val_iou_total += iou
                
        # Tính điểm trung bình của Epoch
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_dice = val_dice_total / len(val_loader)
        avg_val_iou = val_iou_total / len(val_loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | Val IoU: {avg_val_iou:.4f}")

        # 4. Lưu lại mô hình khôn nhất
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), 'models_saved/attention_unet_best.pth')
            print(f"   🌟 Đã lưu mô hình xịn nhất mới! (Dice: {best_dice:.4f})")

    print("🎉 Quá trình huấn luyện đã hoàn tất!")

if __name__ == "__main__":
    train_model()