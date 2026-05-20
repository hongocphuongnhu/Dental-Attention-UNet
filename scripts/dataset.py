import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

class DentalDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.img_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        self.images = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Load ảnh và mask
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png").replace(".jpeg", ".png"))

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 2. Chuẩn hóa pixel về khoảng [0, 1]
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # 3. Thêm chiều kênh (Channel) cho ảnh xám: (256, 256) -> (1, 256, 256)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return torch.from_numpy(image), torch.from_numpy(mask)

# --- Kiểm tra thử bộ nạp dữ liệu ---
if __name__ == "__main__":
    train_ds = DentalDataset("data/processed", split='train')
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    
    imgs, msks = next(iter(train_loader))
    print(f"Batch ảnh shape: {imgs.shape}") # Kỳ vọng: [16, 1, 256, 256]
    print(f"Batch mask shape: {msks.shape}")
    print("Bộ nạp dữ liệu hoạt động hoàn hảo!")