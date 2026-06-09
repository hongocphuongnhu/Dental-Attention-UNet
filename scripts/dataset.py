import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np


class DentalDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.img_dir  = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        self.images   = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        img_path  = os.path.join(self.img_dir, img_name)

        # Tên mask: đổi đuôi .jpg/.jpeg -> .png (theo preprocess.py)
        mask_name = img_name.replace(".jpg", ".png").replace(".jpeg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        # --- Đọc ảnh xám ---
        image = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Kiểm tra file tồn tại (tránh crash âm thầm khi path sai)
        if image is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Không đọc được mask: {mask_path}")

        # --- Chuẩn hóa về [0, 1] ---
        image = image.astype(np.float32) / 255.0
        mask  = mask.astype(np.float32)  / 255.0

        # --- Thêm chiều kênh: (H, W) -> (1, H, W) ---
        image = np.expand_dims(image, axis=0)
        mask  = np.expand_dims(mask,  axis=0)

        return torch.from_numpy(image), torch.from_numpy(mask)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_ds     = DentalDataset("data/processed", split='train')
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    imgs, msks = next(iter(train_loader))
    print(f"Batch ảnh shape : {imgs.shape}")   # [16, 1, 256, 256]
    print(f"Batch mask shape: {msks.shape}")
    print(f"Pixel ảnh  — min: {imgs.min():.2f}, max: {imgs.max():.2f}")
    print(f"Pixel mask — min: {msks.min():.2f}, max: {msks.max():.2f}")
    print("DataLoader hoạt động!")