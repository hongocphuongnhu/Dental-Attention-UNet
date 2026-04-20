
import os
import torch
from torch.utils.data import Dataset
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
        img_path  = os.path.join(self.img_dir, self.images[idx])
        mask_name = os.path.splitext(self.images[idx])[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        image = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype("float32") / 255.0
        mask  = mask.astype("float32")  / 255.0
        image = np.expand_dims(image, axis=0)
        mask  = np.expand_dims(mask,  axis=0)
        return torch.from_numpy(image), torch.from_numpy(mask)
