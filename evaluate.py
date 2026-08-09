import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.dataset import NIHChestXrayDataset, NIH_LABELS
from utils.model import load_trained_densenet121

CSV_TEST_SPLIT_PATH = "models/splits/test_split.csv"
IMG_DIR = "veriseti/images/"
MODEL_PATH = "models/densenet121_kalibre_edilmis_model.pth"
THRESHOLDS_PATH = "models/best_thresholds.json"

RESULTS_JSON_PATH = "models/test_results.json"
RESULTS_MD_PATH = "models/test_results.md"

BATCH_SIZE = 32
NUM_WORKERS = 0
DEFAULT_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_thresholds():
    if not os.path.exists(THRESHOLDS_PATH):
        raise FileNotFoundError(f"{THRESHOLDS_PATH} bulunamadı.")
    with open(THRESHOLDS_PATH, "r") as f:
        return json.load(f)


def get_test_predictions(model, test_loader):
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test seti üzerinde çıkarım (TTA)"):
            images = images.to(DEVICE)
            flipped = torch.flip(images, dims=[3])

            probs_original = torch.sigmoid(model(images))
            probs_flipped = torch.sigmoid(model(flipped))
            probs = ((probs_original + probs_flipped) / 2).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_probs)


def compute_class_metrics(y_true_col, y_prob_col, threshold):
    support = int(y_true_col.sum())

    metrics = {
        "threshold": round(float(threshold), 3),
        "support": support,
        "roc_auc": None,
        "pr_auc": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }

    if 0 < support < len(y_true_col):
        metrics["roc_auc"] = round(float(roc_auc_score(y_true_col, y_prob_col)), 4)
        metrics["pr_auc"] = round(float(average_precision_score(y_true_col, y_prob_col)), 4)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    y_pred_col = (y_prob_col >= threshold).astype(int)
    metrics["precision"] = round(float(precision_score(y_true_col, y_pred_col, zero_division=0)), 4)
    metrics["recall"] = round(float(recall_score(y_true_col, y_pred_col, zero_division=0)), 4)
    metrics["f1"] = round(float(f1_score(y_true_col, y_pred_col, zero_division=0)), 4)

    return metrics


def compute_macro_average(per_class_metrics: dict) -> dict:
    keys = ["roc_auc", "pr_auc", "precision", "recall", "f1"]
    macro = {}
    for key in keys:
        values = [m[key] for m in per_class_metrics.values() if m[key] is not None]
        macro[key] = round(float(np.mean(values)), 4) if values else None
    macro["support"] = sum(m["support"] for m in per_class_metrics.values())
    return macro


def print_results_table(per_class_metrics: dict, macro: dict):
    header = f"{'Sınıf':<20}{'Support':>8}{'ROC-AUC':>10}{'PR-AUC':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}"
    print("\n" + "=" * len(header))
    print("           TEST SETİ ÜZERİNDE NİHAİ SONUÇLAR")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for class_name, m in per_class_metrics.items():
        roc = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "N/A"
        pr = f"{m['pr_auc']:.4f}" if m["pr_auc"] is not None else "N/A"
        print(
            f"{class_name:<20}{m['support']:>8}{roc:>10}{pr:>10}"
            f"{m['precision']:>11.4f}{m['recall']:>9.4f}{m['f1']:>8.4f}"
        )
    print("-" * len(header))
    roc_m = f"{macro['roc_auc']:.4f}" if macro["roc_auc"] is not None else "N/A"
    pr_m = f"{macro['pr_auc']:.4f}" if macro["pr_auc"] is not None else "N/A"
    print(
        f"{'MAKRO ORTALAMA':<20}{macro['support']:>8}{roc_m:>10}{pr_m:>10}"
        f"{macro['precision']:>11.4f}{macro['recall']:>9.4f}{macro['f1']:>8.4f}"
    )
    print("=" * len(header))


def save_markdown_report(per_class_metrics: dict, macro: dict, n_test_patients: int, n_test_images: int):
    lines = [
        "## Test Seti Sonuçları\n",
        f"- Test hasta sayısı: **{n_test_patients}**",
        f"- Test görüntü sayısı: **{n_test_images}**",
        "- Split yöntemi: Hasta bazlı (patient-wise), sızıntısız",
        "- Eşik değerleri: Yalnızca validation setinde F-beta(0.5) taramasıyla belirlendi\n",
        "| Sınıf | Support | ROC-AUC | PR-AUC | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for class_name, m in per_class_metrics.items():
        roc = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "N/A"
        pr = f"{m['pr_auc']:.4f}" if m["pr_auc"] is not None else "N/A"
        lines.append(
            f"| {class_name} | {m['support']} | {roc} | {pr} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |"
        )
    roc_m = f"{macro['roc_auc']:.4f}" if macro["roc_auc"] is not None else "N/A"
    pr_m = f"{macro['pr_auc']:.4f}" if macro["pr_auc"] is not None else "N/A"
    lines.append(
        f"| **Makro Ortalama** | {macro['support']} | {roc_m} | {pr_m} | "
        f"{macro['precision']:.4f} | {macro['recall']:.4f} | {macro['f1']:.4f} |"
    )

    with open(RESULTS_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    print(f"Kullanılan Cihaz: {DEVICE}")

    if not os.path.exists(CSV_TEST_SPLIT_PATH):
        raise FileNotFoundError(f"{CSV_TEST_SPLIT_PATH} bulunamadı.")

    test_df = pd.read_csv(CSV_TEST_SPLIT_PATH)
    test_dataset = NIHChestXrayDataset(test_df, IMG_DIR, NIH_LABELS, transform=val_transforms)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    print("[BİLGİ] Eğitilmiş DenseNet121 modeli yükleniyor...")
    model = load_trained_densenet121(MODEL_PATH, DEVICE)

    thresholds = load_thresholds()

    print("[BİLGİ] Test seti üzerinde tahminler toplanıyor...")
    y_true, y_prob = get_test_predictions(model, test_loader)

    per_class_metrics = {}
    for i, class_name in enumerate(NIH_LABELS):
        threshold = thresholds.get(class_name, DEFAULT_THRESHOLD)
        per_class_metrics[class_name] = compute_class_metrics(y_true[:, i], y_prob[:, i], threshold)

    macro = compute_macro_average(per_class_metrics)

    print_results_table(per_class_metrics, macro)

    os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"per_class": per_class_metrics, "macro_average": macro}, f, indent=2, ensure_ascii=False)

    n_test_patients = test_df["Patient ID"].nunique()
    n_test_images = len(test_df)
    save_markdown_report(per_class_metrics, macro, n_test_patients, n_test_images)

    print(f"\n[BİLGİ] Sonuçlar kaydedildi:")
    print(f"  - {RESULTS_JSON_PATH}")
    print(f"  - {RESULTS_MD_PATH}")


if __name__ == "__main__":
    main()