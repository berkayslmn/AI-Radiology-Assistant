import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

NIH_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]


class NIHDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')

        labels_str = self.data_frame.iloc[idx]['Finding Labels']
        labels_list = labels_str.split('|')

        label_tensor = torch.zeros(len(NIH_LABELS), dtype=torch.float32)
        for i, label in enumerate(NIH_LABELS):
            if label in labels_list:
                label_tensor[i] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, label_tensor