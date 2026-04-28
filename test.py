"""
Created on Fri Mar 20 2026

@author: Santosh Prakash
Evaluation / Testing script for HybridSegNet on BUSI dataset
"""

import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.hybridsegnet import HybridSegNet
from datasets.busi_dataset import BUSIDataset
from losses.hybrid_loss import HybridLoss
from utils.metrics import dice_score, iou_score


# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
MODEL_PATH = r"C:\Users\USER\Desktop\Breast_cancer\HybridSegNet-main\Final_model\cv_runs_exp1\single_cv_run\fold_5\best_model.pth"

TEST_IMG_DIR = r"C:\Users\USER\Desktop\Breast_cancer\HybridSegNet-main\Final_model\data\train\images"
TEST_MASK_DIR = r"C:\Users\USER\Desktop\Breast_cancer\HybridSegNet-main\Final_model\data\train\masks"

SAVE_PREDICTIONS = True
PRED_SAVE_DIR = "test_predictions"
THRESHOLD = 0.1


# =========================
# Utility Metrics
# =========================
def pixel_accuracy(pred, mask):
    """
    pred, mask: tensors of shape [B, 1, H, W]
    pred should already be binary (0/1)
    """
    correct = (pred == mask).float().sum()
    total = torch.numel(pred)
    return correct / total


def precision_score(pred, mask, eps=1e-7):
    pred = pred.float()
    mask = mask.float()

    tp = (pred * mask).sum()
    fp = (pred * (1 - mask)).sum()

    return (tp + eps) / (tp + fp + eps)


def recall_score(pred, mask, eps=1e-7):
    pred = pred.float()
    mask = mask.float()

    tp = (pred * mask).sum()
    fn = ((1 - pred) * mask).sum()

    return (tp + eps) / (tp + fn + eps)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# =========================
# Dataset / Loader
# =========================
test_dataset = BUSIDataset(
    TEST_IMG_DIR,
    TEST_MASK_DIR
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================
# Model / Loss
# =========================
model = HybridSegNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

criterion = HybridLoss()


# =========================
# Prepare Save Folder
# =========================
if SAVE_PREDICTIONS:
    ensure_dir(PRED_SAVE_DIR)


# =========================
# Evaluation Loop
# =========================
test_loss = 0.0
dice_avg = 0.0
iou_avg = 0.0
acc_avg = 0.0
prec_avg = 0.0
rec_avg = 0.0

with torch.no_grad():
    for idx, (img, mask) in enumerate(tqdm(test_loader, desc="Testing")):
        img = img.to(DEVICE)
        mask = mask.to(DEVICE)

        # Forward
        pred = model(img)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=10.0, neginf=-10.0)

        # Loss
        loss = criterion(pred, mask)
        test_loss += loss.item()

        # Probability + Binary mask
        pred_sig = torch.sigmoid(pred)
        pred_bin = (pred_sig > THRESHOLD).float()

        # Metrics
        dice_avg += dice_score(pred_sig, mask).item()
        iou_avg += iou_score(pred_sig, mask).item()
        acc_avg += pixel_accuracy(pred_bin, mask).item()
        prec_avg += precision_score(pred_bin, mask).item()
        rec_avg += recall_score(pred_bin, mask).item()

        # Save predicted mask
        if SAVE_PREDICTIONS:
            save_path = os.path.join(PRED_SAVE_DIR, f"pred_{idx:03d}.png")
            save_image(pred_bin, save_path)

            # Optional: save probability map too
            prob_path = os.path.join(PRED_SAVE_DIR, f"prob_{idx:03d}.png")
            save_image(pred_sig, prob_path)

# =========================
# Final Results
# =========================
num_samples = len(test_loader)

test_loss /= num_samples
dice_avg /= num_samples
iou_avg /= num_samples
acc_avg /= num_samples
prec_avg /= num_samples
rec_avg /= num_samples

print("\n================ TEST RESULTS ================")
print(f"Test Loss      : {test_loss:.4f}")
print(f"Dice Score     : {dice_avg:.4f}")
print(f"IoU Score      : {iou_avg:.4f}")
print(f"Pixel Accuracy : {acc_avg:.4f}")
print(f"Precision      : {prec_avg:.4f}")
print(f"Recall         : {rec_avg:.4f}")
print("==============================================")