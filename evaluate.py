import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

CSV_PATH = "veriseti/Filtered_Data_Entry.csv"
IMG_DIR = "veriseti/images/"
MODEL_PATH = "models/resnet18_kalibre_edilmis_model.pth"
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

    print("\n[BILGI] En iyi model ağırlıkları yükleniyor...")
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    print("[BILGI] Doğrulama verisi üzerinde değerlendirme başlatıldı...\n")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Röntgenler Analiz Ediliyor"):
            images = images.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_preds.extend(probs)
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    print("\n" + "=" * 45)
    print("      SINIF BAZINDA ROC-AUC BAŞARI SKORLARI")
    print("=" * 45)

    for i, class_name in enumerate(CLASS_NAMES):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            print(f"{class_name:<20}: {auc:.4f}")
        except ValueError:
            print(f"{class_name:<20}: Hesaplanamadı (Veri yetersiz)")

    print("=" * 45)
    print("NOT: ROC-AUC skoru;")
    print(" 0.5 -> Tamamen rastgele tahmin (Başarısız)")
    print(" 0.7 - 0.8 -> Kabul edilebilir düzeyde ayrım")
    print(" 0.8 - 0.9 -> Başarılı model")
    print(" 1.0 -> Kusursuz tespit yeteneği")