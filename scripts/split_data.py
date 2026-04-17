import os
import shutil
import random

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.model_selection import train_test_split

def split_data():
    # 1. Đường dẫn gốc
    data_path = "data/processed"
    img_dir = os.path.join(data_path, "images")
    mask_dir = os.path.join(data_path, "masks")
    
    # Lấy danh sách file và sắp xếp để khớp nhau
    images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    masks = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png'))])

    # 2. Chia tỉ lệ: 80% Train - 10% Val - 10% Test
    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42)

    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs, temp_masks, test_size=0.5, random_state=42)

    # 3. Định nghĩa các thư mục đích
    splits = {
        'train': (train_imgs, train_masks),
        'val': (val_imgs, val_masks),
        'test': (test_imgs, test_masks)
    }

    for split_name, (imgs, msks) in splits.items():
        # Tạo thư mục con: data/processed/train/images, v.v...
        os.makedirs(os.path.join(data_path, split_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(data_path, split_name, "masks"), exist_ok=True)

        for img, msk in zip(imgs, msks):
            # Di chuyển file từ folder gốc vào folder split
            shutil.move(os.path.join(img_dir, img), os.path.join(data_path, split_name, "images", img))
            shutil.move(os.path.join(mask_dir, msk), os.path.join(data_path, split_name, "masks", msk))

    print(f"✅ Đã chia xong 1500 ảnh!")
    print(f"📂 Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

if __name__ == "__main__":
    split_data()