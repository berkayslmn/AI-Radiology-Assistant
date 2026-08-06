import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve

CSV_PATH = "veriseti/Filtered_Data_Entry.csv"
IMG_DIR = "veriseti/images/"
MODEL_PATH = "models/densenet121_kalibre_edilmis_model.pth"
THRESHOLD_SAVE_PATH = "models/best_thresholds.json"
BATCH_SIZE = 32
NUM_CLASSES = 14

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
               'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
               'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class NIHChestXrayDataset(Dataset):
    def __init__(self, dataframe, img_dir, class_names, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.class_names = class_names

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['Image Index'])
        image = Image.open(img_name).convert('RGB')

        labels_str = self.dataframe.iloc[idx]['Finding Labels']
        labels = torch.zeros(len(self.class_names), dtype=torch.float32)
        if labels_str != 'No Finding':
            for i, c in enumerate(self.class_names):
                if c in labels_str:
                    labels[i] = 1.0

        if self.transform:
            image = self.transform(image)
        return image, labels


if __name__ == '__main__':
    print(f"Kullanılan Cihaz: {device}")

    df = pd.read_csv(CSV_PATH)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    val_dataset = NIHChestXrayDataset(val_df, IMG_DIR, CLASS_NAMES, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Model yükleniyor...")
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    print("Doğrulama seti üzerinden tahminler toplanıyor (Bu işlem birkaç dakika sürebilir)...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="İşleniyor"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    print("\nHer sınıf için optimum eşik değerleri hesaplanıyor...")
    best_thresholds = {}

    for i, class_name in enumerate(CLASS_NAMES):
        true_labels = all_labels[:, i]
        predictions = all_preds[:, i]

        precision, recall, thresholds = precision_recall_curve(true_labels, predictions)

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        best_thresholds[class_name] = float(best_thresh)
        print(f"{class_name.ljust(20)}: Optimum Eşik = % {best_thresh * 100:.2f}")

    with open(THRESHOLD_SAVE_PATH, "w") as f:
        json.dump(best_thresholds, f, indent=4)

    print(f"\n>>> Tüm eşik değerleri başarıyla {THRESHOLD_SAVE_PATH} dosyasına kaydedildi!")
