import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse

from models.attention_unet import AttentionUNet
from scripts.dataset import DentalDataset
from scripts.metrics import get_metrics


# ---------------------------------------------------------------------------
# COMBO LOSS: BCEWithLogitsLoss + Dice Loss
# ---------------------------------------------------------------------------
# Tại sao kết hợp 2 loss?
#   - BCEWithLogitsLoss: đánh giá từng pixel độc lập, ổn định số học
#     (tích hợp sigmoid bên trong bằng công thức log-sum-exp, tránh NaN)
#   - Dice Loss: đánh giá độ chồng khít toàn vùng, chống class imbalance
#     (vùng nền ~95% pixel >> vùng tổn thương ~5%)
# ---------------------------------------------------------------------------
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # BCEWithLogitsLoss nhận LOGIT thô, tự áp sigmoid bên trong
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1e-6):
        # --- Phần BCE (nhận logit trực tiếp) ---
        bce_loss = self.bce(inputs, targets)

        # --- Phần Dice (cần xác suất, phải áp sigmoid thủ công) ---
        inputs_prob = torch.sigmoid(inputs)
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (
            inputs_flat.sum() + targets_flat.sum() + smooth
        )

        return bce_loss + dice_loss


# ---------------------------------------------------------------------------
# THAM SỐ DÒNG LỆNH
# ---------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình phân đoạn tổn thương nha khoa trên X-quang"
    )
    parser.add_argument('--arch',       type=str,   default='attention_unet',
                        choices=['attention_unet', 'vgg_unet', 'res_unet'])
    parser.add_argument('--save_name',  type=str,   default='attention_unet_best.pth')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--data_dir',   type=str,   default='data/processed')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# VÒNG LẶP TRAIN CHÍNH
# ---------------------------------------------------------------------------
def train_model():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs("models_saved", exist_ok=True)

    print(f"Thiết bị   : {device}")
    print(f"Kiến trúc  : {args.arch.upper()}")
    print(f"Lưu model  : models_saved/{args.save_name}")
    print("-" * 50)

    # --- Dữ liệu ---
    train_ds     = DentalDataset(root_dir=args.data_dir, split='train')
    val_ds       = DentalDataset(root_dir=args.data_dir, split='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Train: {len(train_ds)} ảnh | Val: {len(val_ds)} ảnh")

    # --- Mô hình ---
    if args.arch == 'attention_unet':
        model = AttentionUNet(n_classes=1).to(device)
    elif args.arch == 'res_unet':
        from models.res_unet import ResUNet
        model = ResUNet(n_classes=1).to(device)
    else:
        raise NotImplementedError(f"Kiến trúc '{args.arch}' chưa được implement.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tham số    : {total_params:,}")
    print("-" * 50)

    # --- Loss, Optimizer ---
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_dice = 0.0

    for epoch in range(args.epochs):

        # ── TRAIN ──────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)          # logit thô
            loss    = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # ── VALIDATION ─────────────────────────────────────────────────────
        model.eval()
        val_dice_total = 0.0
        val_iou_total  = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)      # logit thô
                dice, iou = get_metrics(outputs, masks)
                val_dice_total += dice
                val_iou_total  += iou

        avg_loss = epoch_loss     / len(train_loader)
        avg_dice = val_dice_total / len(val_loader)
        avg_iou  = val_iou_total  / len(val_loader)

        print(
            f"Epoch [{epoch+1:02d}/{args.epochs}] "
            f"| Loss: {avg_loss:.4f} "
            f"| Val Dice: {avg_dice:.4f} "
            f"| Val IoU: {avg_iou:.4f}"
        )

        # ── LƯU MODEL TỐT NHẤT ─────────────────────────────────────────────
        if avg_dice > best_dice:
            best_dice = avg_dice
            save_path = os.path.join("models_saved", args.save_name)
            torch.save(model.state_dict(), save_path)
            print(f"           Saved best model — Dice: {best_dice:.4f}")

    print("=" * 50)
    print(f"Hoàn tất! Best Val Dice: {best_dice:.4f}")
    print(f"Model đã lưu tại: models_saved/{args.save_name}")


if __name__ == "__main__":
    train_model()