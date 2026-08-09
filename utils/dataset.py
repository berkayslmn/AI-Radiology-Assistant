import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

NUM_CLASSES = len(NIH_LABELS)


class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, class_names=None, transform=None):
        if isinstance(csv_file, pd.DataFrame):
            self.dataframe = csv_file.reset_index(drop=True)
        else:
            self.dataframe = pd.read_csv(csv_file)

        self.img_dir = img_dir
        self.transform = transform
        self.class_names = class_names if class_names is not None else NIH_LABELS

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_name = os.path.join(self.img_dir, row["Image Index"])
        image = Image.open(img_name).convert("RGB")

        label_tensor = torch.zeros(len(self.class_names), dtype=torch.float32)
        labels_str = row["Finding Labels"]

        if isinstance(labels_str, str) and labels_str != "No Finding":
            found_labels = set(labels_str.split("|"))
            for i, class_name in enumerate(self.class_names):
                if class_name in found_labels:
                    label_tensor[i] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, label_tensor


NIHDataset = NIHChestXrayDataset