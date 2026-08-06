import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import DenseNet121_Weights
from PIL import Image
from tqdm import tqdm

CSV_PATH = "veriseti/Filtered_Data_Entry.csv"
IMG_DIR = "veriseti/images/"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_CLASSES = 14
EARLY_STOPPING_PATIENCE = 4

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
               'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
               'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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


def get_pos_weights(df, class_names):
    pos_weights = []
    total_samples = len(df)

    for c in class_names:
        pos_count = df['Finding Labels'].str.contains(c, regex=False).sum()
        neg_count = total_samples - pos_count
        weight = neg_count / (pos_count + 1e-7)
        pos_weights.append(weight)

    return torch.tensor(pos_weights, dtype=torch.float32)


if __name__ == '__main__':
    print(f"Kullanılan Cihaz: {device}")

    df = pd.read_csv(CSV_PATH)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    train_dataset = NIHChestXrayDataset(train_df, IMG_DIR, CLASS_NAMES, transform=train_transforms)
    val_dataset = NIHChestXrayDataset(val_df, IMG_DIR, CLASS_NAMES, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    pos_weight_tensor = get_pos_weights(train_df, CLASS_NAMES).to(device)

    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    patience_counter = 0

    os.makedirs("models", exist_ok=True)
    MODEL_SAVE_PATH = "models/densenet121_kalibre_edilmis_model.pth"

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 15)

        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader, desc="Eğitim"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        epoch_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Doğrulama"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)

        print(f"Eğitim Kaybı (Train Loss): {epoch_train_loss:.4f} | Doğrulama Kaybı (Val Loss): {epoch_val_loss:.4f}")

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f">>> Yeni en iyi model kaydedildi! ({MODEL_SAVE_PATH})")
        else:
            patience_counter += 1
            print(
                f"Uyarı: Doğrulama kaybı iyileşmedi. Erken durdurma sayacı: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("\n[!] Aşırı öğrenme (Overfitting) tespit edildi. Erken durdurma (Early Stopping) tetiklendi!")
                break

    print("\nEğitim süreci sonlandırıldı.")