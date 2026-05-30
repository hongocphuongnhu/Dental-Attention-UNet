"""
evaluate.py — Đánh giá mô hình trên tập Test và xuất ảnh so sánh.

Cách chạy:
    python evaluate.py --model_path models_saved/attention_unet_best.pth

Kết quả xuất ra:
    results/
        metrics_report.txt        ← Dice & IoU trung bình toàn tập test
        sample_0.png              ← Ảnh so sánh: Gốc | Mask bác sĩ | Mask AI
        sample_1.png
        ...
"""

import torch
import cv2
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader

from models.attention_unet import AttentionUNet
from scripts.dataset import DentalDataset
from scripts.metrics import get_metrics


# ---------------------------------------------------------------------------
# THAM SỐ DÒNG LỆNH
# ---------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình trên tập test")
    parser.add_argument('--model_path', type=str,
                        default='models_saved/attention_unet_best.pth',
                        help='Đường dẫn đến file .pth đã lưu')
    parser.add_argument('--data_dir',   type=str,   default='data/processed')
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--threshold',  type=float, default=0.5,
                        help='Ngưỡng sigmoid để quyết định pixel là tổn thương')
    parser.add_argument('--num_samples', type=int,  default=10,
                        help='Số ảnh so sánh muốn lưu ra')
    parser.add_argument('--output_dir', type=str,   default='results')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# HÀM VẼ ẢNH SO SÁNH 3 CỘT: Ảnh gốc | Mask bác sĩ | Mask AI
# ---------------------------------------------------------------------------
def save_comparison(img_tensor, mask_tensor, pred_tensor, save_path, threshold=0.5):
    """
    img_tensor  : (1, H, W) float [0,1]
    mask_tensor : (1, H, W) float [0,1]
    pred_tensor : (1, H, W) float — logit thô từ model
    """
    # Chuyển về numpy uint8
    img  = (img_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    mask = (mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

    # Logit -> sigmoid -> nhị phân
    pred_prob = torch.sigmoid(pred_tensor).squeeze().cpu().numpy()
    pred_bin  = ((pred_prob > threshold) * 255).astype(np.uint8)

    # Chuyển về ảnh màu để ghép cạnh nhau
    img_color  = cv2.cvtColor(img,      cv2.COLOR_GRAY2BGR)
    mask_color = cv2.cvtColor(mask,     cv2.COLOR_GRAY2BGR)
    pred_color = cv2.cvtColor(pred_bin, cv2.COLOR_GRAY2BGR)

    # Overlay: tô màu xanh lên vùng AI predict đúng, đỏ cho sai
    overlay = img_color.copy()
    gt   = mask > 127
    pred = pred_bin > 127
    overlay[pred & gt,  1] = 200   # Xanh lá = đúng (True Positive)
    overlay[pred & ~gt, 2] = 200   # Đỏ = dư (False Positive)
    overlay[~pred & gt, 0] = 200   # Xanh dương = bỏ sót (False Negative)

    # Thêm tiêu đề
    h = img_color.shape[0]
    label_h = 30
    def labeled(panel, text):
        canvas = np.zeros((h + label_h, panel.shape[1], 3), dtype=np.uint8)
        canvas[label_h:] = panel
        cv2.putText(canvas, text, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        return canvas

    col1 = labeled(img_color,  "Input X-ray")
    col2 = labeled(mask_color, "Ground truth (bac si)")
    col3 = labeled(pred_color, f"AI predict (t={threshold})")
    col4 = labeled(overlay,    "Overlay: Green=TP Red=FP Blue=FN")

    combined = np.hstack([col1, col2, col3, col4])
    cv2.imwrite(save_path, combined)


# ---------------------------------------------------------------------------
# HÀM ĐÁNH GIÁ CHÍNH
# ---------------------------------------------------------------------------
def evaluate():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Thiết bị   : {device}")
    print(f"Model      : {args.model_path}")
    print(f"Ngưỡng     : {args.threshold}")
    print("-" * 50)

    # --- Tải model ---
    if 'res_unet' in args.model_path:
        from models.res_unet import ResUNet
        model = ResUNet(n_classes=1).to(device)
    else:
        model = AttentionUNet(n_classes=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # --- Tập test ---
    test_ds     = DentalDataset(root_dir=args.data_dir, split='test')
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Tập test   : {len(test_ds)} ảnh ({len(test_loader)} batch)")
    print("-" * 50)

    # --- Vòng lặp đánh giá ---
    total_dice = 0.0
    total_iou  = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks  = masks.to(device)
            preds  = model(images)   # logit thô

            # Tính metrics cho cả batch
            dice, iou = get_metrics(preds, masks)
            total_dice += dice
            total_iou  += iou

            # Lưu ảnh so sánh cho một số mẫu đầu tiên
            for i in range(images.size(0)):
                if sample_count >= args.num_samples:
                    break
                save_path = os.path.join(args.output_dir, f"sample_{sample_count:02d}.png")
                save_comparison(
                    images[i], masks[i], preds[i],
                    save_path, threshold=args.threshold
                )
                sample_count += 1

    # --- Tính trung bình ---
    avg_dice = total_dice / len(test_loader)
    avg_iou  = total_iou  / len(test_loader)

    # --- In kết quả ---
    print(f"Dice Score trung bình : {avg_dice:.4f}  ({avg_dice*100:.2f}%)")
    print(f"IoU trung bình        : {avg_iou:.4f}  ({avg_iou*100:.2f}%)")
    print("-" * 50)

    # Nhận xét tự động
    if avg_dice >= 0.90:
        verdict = "Xuất sắc — model phân đoạn rất tốt."
    elif avg_dice >= 0.80:
        verdict = "Tốt — đạt yêu cầu nghiên cứu."
    elif avg_dice >= 0.70:
        verdict = "Khá — có thể cải thiện thêm."
    else:
        verdict = "Cần xem lại — thử tăng epoch hoặc điều chỉnh learning rate."
    print(f"Nhận xét   : {verdict}")

    # --- Lưu báo cáo ---
    report_path = os.path.join(args.output_dir, "metrics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== BÁO CÁO ĐÁNH GIÁ MÔ HÌNH ===\n\n")
        f.write(f"Model      : {args.model_path}\n")
        f.write(f"Tập test   : {len(test_ds)} ảnh\n")
        f.write(f"Ngưỡng     : {args.threshold}\n\n")
        f.write(f"Dice Score : {avg_dice:.4f}\n")
        f.write(f"IoU        : {avg_iou:.4f}\n\n")
        f.write(f"Nhận xét   : {verdict}\n")
    print(f"\nĐã lưu báo cáo  : {report_path}")
    print(f"Đã lưu {sample_count} ảnh so sánh vào thư mục: {args.output_dir}/")


if __name__ == "__main__":
    evaluate()