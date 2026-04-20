import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append('.')

from dataset import DentalDataset
from model   import ResUNet
from utils   import bce_dice_loss, dice_score

DATA_PATH  = "/content/data/processed"
EPOCHS     = 50
BATCH_SIZE = 16
LR         = 1e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH  = "/content/drive/MyDrive/best_model_resunet.pth"

print(f" Training trên: {DEVICE.upper()}")

train_ds = DentalDataset(DATA_PATH, split='train')
val_ds   = DentalDataset(DATA_PATH, split='val')

pin = DEVICE == "cuda"
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=pin)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

print(f"Train: {len(train_ds)} ảnh | Val: {len(val_ds)} ảnh")

model     = ResUNet().to(DEVICE)

optimizer = Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    model.train()
    t_loss, t_dice = 0, 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = bce_dice_loss(preds, masks)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        t_dice += dice_score(preds.detach(), masks)

    model.eval()
    v_loss, v_dice = 0, 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds  = model(imgs)
            v_loss += bce_dice_loss(preds, masks).item()
            v_dice += dice_score(preds, masks)

    n_t = len(train_loader)
    n_v = len(val_loader)
    scheduler.step(v_loss / n_v)

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train Loss: {t_loss/n_t:.4f}  Dice: {t_dice/n_t:.4f} | "
          f"Val Loss: {v_loss/n_v:.4f}  Dice: {v_dice/n_v:.4f}")

    if v_loss / n_v < best_val_loss:
        best_val_loss = v_loss / n_v
        torch.save(model.state_dict(), SAVE_PATH)
        print(f" Saved best model!")

print("\n Training hoàn tất! Model lưu tại:", SAVE_PATH)
