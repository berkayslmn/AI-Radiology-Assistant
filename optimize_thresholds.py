import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, fbeta_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.dataset import NIHChestXrayDataset, NIH_LABELS
from utils.model import load_trained_densenet121

CSV_VAL_SPLIT_PATH = "models/splits/val_split.csv"
IMG_DIR = "veriseti/images/"
MODEL_PATH = "models/densenet121_kalibre_edilmis_model.pth"
THRESHOLDS_SAVE_PATH = "models/best_thresholds.json"

BATCH_SIZE = 32
NUM_WORKERS = 0
DEFAULT_THRESHOLD = 0.5

FBETA_BETA = 0.5

THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01

MIN_SUPPORT_FOR_TUNING = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_val_predictions(model, val_loader):
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation üzerinde çıkarım (TTA)"):
            images = images.to(DEVICE)
            flipped = torch.flip(images, dims=[3])

            probs_original = torch.sigmoid(model(images))
            probs_flipped = torch.sigmoid(model(flipped))
            probs = ((probs_original + probs_flipped) / 2).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_probs)


def find_best_threshold_for_class(y_true_col, y_prob_col):
    support = int(y_true_col.sum())

    if support == 0:
        return DEFAULT_THRESHOLD, 0.0, support, "sabit (pozitif örnek yok)"

    if support < MIN_SUPPORT_FOR_TUNING:
        y_pred_col = (y_prob_col >= DEFAULT_THRESHOLD).astype(int)
        score = fbeta_score(y_true_col, y_pred_col, beta=FBETA_BETA, zero_division=0)
        return DEFAULT_THRESHOLD, score, support, "sabit (az örnek)"

    best_threshold = DEFAULT_THRESHOLD
    best_score = -1.0

    for threshold in np.arange(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP):
        y_pred_col = (y_prob_col >= threshold).astype(int)
        score = fbeta_score(y_true_col, y_pred_col, beta=FBETA_BETA, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    y_pred_at_best = (y_prob_col >= best_threshold).astype(int)
    if y_pred_at_best.sum() == 0:
        fallback_threshold = DEFAULT_THRESHOLD
        fallback_f1 = -1.0
        for threshold in np.arange(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP):
            y_pred_col = (y_prob_col >= threshold).astype(int)
            f1 = f1_score(y_true_col, y_pred_col, zero_division=0)
            if f1 > fallback_f1:
                fallback_f1 = f1
                fallback_threshold = float(threshold)
        return fallback_threshold, fallback_f1, support, "F1-fallback (F-beta sınıfı susturdu)"

    return best_threshold, best_score, support, f"F{FBETA_BETA}-tarama"


def main():
    print(f"Kullanılan Cihaz: {DEVICE}")

    if not os.path.exists(CSV_VAL_SPLIT_PATH):
        raise FileNotFoundError(f"{CSV_VAL_SPLIT_PATH} bulunamadı.")

    val_df = pd.read_csv(CSV_VAL_SPLIT_PATH)
    val_dataset = NIHChestXrayDataset(val_df, IMG_DIR, NIH_LABELS, transform=val_transforms)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    print("[BİLGİ] Eğitilmiş DenseNet121 modeli yükleniyor...")
    model = load_trained_densenet121(MODEL_PATH, DEVICE)

    print("[BİLGİ] Validation seti üzerinde tahminler toplanıyor...")
    y_true, y_prob = get_val_predictions(model, val_loader)

    print("\n" + "=" * 70)
    print(f"   SINIF BAZINDA OPTİMUM EŞİK DEĞERLERİ (Validation, F{FBETA_BETA}-Score)")
    print("=" * 70)

    best_thresholds = {}
    for i, class_name in enumerate(NIH_LABELS):
        threshold, score, support, method_tag = find_best_threshold_for_class(y_true[:, i], y_prob[:, i])
        best_thresholds[class_name] = round(threshold, 3)

        print(
            f"{class_name:<20}: eşik={threshold:.2f}  (skor={score:.4f}, "
            f"support={support:>3}, yöntem={method_tag})"
        )

    print("=" * 70)

    os.makedirs(os.path.dirname(THRESHOLDS_SAVE_PATH), exist_ok=True)
    with open(THRESHOLDS_SAVE_PATH, "w") as f:
        json.dump(best_thresholds, f, indent=2, ensure_ascii=False)

    print(f"\n[BİLGİ] Eşik değerleri kaydedildi: {THRESHOLDS_SAVE_PATH}")


if __name__ == "__main__":
    main()
