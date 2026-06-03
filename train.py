import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import csv
import argparse

from models.attention_unet import AttentionUNet
from scripts.dataset import DentalDataset
from scripts.metrics import get_metrics


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1e-6):
        bce_loss = self.bce(inputs, targets)
        inputs_prob = torch.sigmoid(inputs)
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (
            inputs_flat.sum() + targets_flat.sum() + smooth
        )
        return bce_loss + dice_loss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',       type=str,   default='attention_unet',
                        choices=['attention_unet', 'vgg_unet', 'res_unet'])
    parser.add_argument('--save_name',  type=str,   default='attention_unet_best.pth')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--data_dir',   type=str,   default='data/processed')
    return parser.parse_args()


def train_model():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs("models_saved", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print(f"Thiết bị   : {device}")
    print(f"Kiến trúc  : {args.arch.upper()}")
    print(f"Lưu model  : models_saved/{args.save_name}")
    print("-" * 60)

    train_ds     = DentalDataset(root_dir=args.data_dir, split='train')
    val_ds       = DentalDataset(root_dir=args.data_dir, split='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Train: {len(train_ds)} anh | Val: {len(val_ds)} anh")

    if args.arch == 'attention_unet':
        model = AttentionUNet(n_classes=1).to(device)
    else:
        raise NotImplementedError(f"Kien truc '{args.arch}' chua duoc implement.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tham so    : {total_params:,}")
    print("-" * 60)

    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_dice = 0.0

    log_path = os.path.join("logs", args.save_name.replace(".pth", "_log.csv"))
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "val_dice", "val_iou"])
    print(f"Log CSV    : {log_path}")
    print("=" * 60)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        val_dice_total = 0.0
        val_iou_total  = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
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
            f"| Val IoU:  {avg_iou:.4f}"
        )

        log_writer.writerow([epoch + 1, round(avg_loss, 6),
                             round(avg_dice, 6), round(avg_iou, 6)])
        log_file.flush()

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), os.path.join("models_saved", args.save_name))
            print(f"           Saved best model - Dice: {best_dice:.4f}")

    log_file.close()
    print("=" * 60)
    print(f"Hoan tat! Best Val Dice : {best_dice:.4f}")
    print(f"Log CSV    : {log_path}")


if __name__ == "__main__":
    train_model()