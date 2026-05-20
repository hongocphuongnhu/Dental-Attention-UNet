import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse


from models.attention_unet import AttentionUNet        

from scripts.dataset import DentalDataset 
from scripts.metrics import get_metrics

# --- 2. ĐỊNH NGHĨA COMBO LOSS (BCE + DICE) ---
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss() 

    def forward(self, inputs, targets, smooth=1e-6):
        # Tính BCE: Tập trung vào độ chính xác từng pixel
        bce_loss = self.bce(inputs, targets)
        
        # Tính Dice: Tập trung vào độ khớp của vùng răng
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth))
        
        return bce_loss + dice_loss

# --- 3. THIẾT LẬP THAM SỐ LỆNH ---
def get_args():
    parser = argparse.ArgumentParser(description="Chương trình huấn luyện mô hình phân đoạn tổn thương nha khoa trên ảnh X-quang")
    parser.add_argument('--arch', type=str, default='attention_unet', 
                        choices=['attention_unet', 'vgg_unet', 'res_unet'])
    parser.add_argument('--save_name', type=str, default='model_best.pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    return parser.parse_args()

# --- 4. HÀM TRAIN CHÍNH ---
def train_model():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs("models_saved", exist_ok=True)

    print(f"Kiến trúc: {args.arch.upper()} | Lưu tại: {args.save_name}")

    # Khởi tạo Data
    data_dir = "data/processed"
    train_ds = DentalDataset(root_dir=data_dir, split='train')
    val_ds = DentalDataset(root_dir=data_dir, split='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Khởi tạo mô hình theo lựa chọn của từng người
    if args.arch == 'attention_unet':
        model = AttentionUNet(n_classes=1).to(device)
    elif args.arch == 'vgg_unet':
        # model = VGGUNet(n_classes=1).to(device)
        print("Đang khởi tạo VGG-UNet...")
        pass
    elif args.arch == 'res_unet':
        # model = ResNetUNet(n_classes=1).to(device)
        print("Đang khởi tạo Res-UNet...")
        pass

    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_dice = 0.0

    for epoch in range(args.epochs):
        # -- TRAIN --
        model.train()
        epoch_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # -- VALIDATION --
        model.eval()
        val_dice_total, val_iou_total = 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                dice, iou = get_metrics(outputs, masks)
                val_dice_total += dice
                val_iou_total += iou
                
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_dice = val_dice_total / len(val_loader)
        avg_val_iou = val_iou_total / len(val_loader)

        print(f"Epoch [{epoch+1}/{args.epochs}] | Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | Val IoU: {avg_val_iou:.4f}")

        # Lưu lại mô hình tốt nhất
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join("models_saved", args.save_name))
            print(f"Saved Best Model: {best_dice:.4f}")

    print("Hoàn tất!")

if __name__ == "__main__":
    train_model()