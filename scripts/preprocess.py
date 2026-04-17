import cv2
import os

def process_data():
    raw_img_path = "data/raw/images"
    raw_mask_path = "data/raw/masks"  
    proc_img_path = "data/processed/images"
    proc_mask_path = "data/processed/masks"
    
    os.makedirs(proc_img_path, exist_ok=True)
    os.makedirs(proc_mask_path, exist_ok=True)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    files = [f for f in os.listdir(raw_img_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    total = len(files)
    
    print(f"Bắt đầu xử lý {total} tấm ảnh... Core i9 đang tăng tốc!")

    for i, filename in enumerate(files):
        # --- Xử lý Ảnh gốc ---
        img = cv2.imread(os.path.join(raw_img_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        img = cv2.resize(img, (256, 256))
        img = clahe.apply(img)
        # Lưu ảnh gốc (giữ nguyên tên và đuôi của nó)
        cv2.imwrite(os.path.join(proc_img_path, filename), img)
        
        # --- Xử lý Mask ---
        mask_name_no_ext = os.path.splitext(filename)[0]
        mask_file = mask_name_no_ext + ".bmp"
        mask_full_path = os.path.join(raw_mask_path, mask_file)

        if os.path.exists(mask_full_path):
            mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            # Chuyển về đen trắng tuyệt đối (0 và 255)
            _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
            
            # ĐỔI ĐUÔI SANG .png KHI LƯU CHO NHẸ VÀ ĐỒNG BỘ
            save_mask_name = mask_name_no_ext + ".png"
            cv2.imwrite(os.path.join(proc_mask_path, save_mask_name), mask)
        else:
            print(f"\n⚠️ Thiếu mask cho: {filename}")

        # In tiến độ ra màn hình sau mỗi 100 ảnh để không bị spam quá nhiều
        if (i + 1) % 100 == 0:
            print(f"Đã xong: {i + 1}/{total} ảnh...")

    print("\nHoàn thành! Dữ liệu đã sẵn sàng.")

if __name__ == "__main__":
    process_data()