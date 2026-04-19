import torch

def get_metrics(y_pred, y_true, smooth=1e-6):
    """
    Hàm tính toán Dice Score và IoU.
    y_pred: Ảnh kết quả do AI vẽ (đã qua sigmoid, giá trị từ 0-1)
    y_true: Ảnh Mask gốc của bác sĩ
    """
    # Ép kiểu dự đoán về 0 (Nền) hoặc 1 (Răng) với ngưỡng 0.5
    y_pred = (y_pred > 0.5).float()
    y_true = y_true.float()

    # Tính phần diện tích giao nhau (Intersection)
    intersection = (y_pred * y_true).sum()
    
    # Công thức Dice Score
    dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    
    # Công thức IoU
    iou = (intersection + smooth) / (y_pred.sum() + y_true.sum() - intersection + smooth)
    
    return dice.item(), iou.item()