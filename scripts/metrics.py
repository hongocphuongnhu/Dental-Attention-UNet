import torch


def get_metrics(y_pred, y_true, smooth=1e-6):
    """
    Tính Dice Score và IoU cho bài toán binary segmentation.

    Args:
        y_pred : output thô từ model (logit, CHƯA qua sigmoid)
        y_true : mask ground truth (giá trị 0 hoặc 1)
        smooth : hệ số làm mịn, tránh chia cho 0

    Returns:
        (dice, iou) — tuple float trong khoảng [0, 1]
    """
    # Bước 1: Logit -> xác suất -> nhị phân (0 hoặc 1)
    # Phải áp sigmoid trước vì model không còn sigmoid ở output
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    y_true = y_true.float()

    # Bước 2: Tính phần giao nhau (Intersection)
    intersection = (y_pred * y_true).sum()

    # Dice Score: 2|A∩B| / (|A| + |B|)
    dice = (2.0 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

    # IoU: |A∩B| / |A∪B|
    iou = (intersection + smooth) / (y_pred.sum() + y_true.sum() - intersection + smooth)

    return dice.item(), iou.item()