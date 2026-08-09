import pandas as pd

NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

paths = {
    "TRAIN": "models/splits/train_split.csv",
    "VAL": "models/splits/val_split.csv",
    "TEST": "models/splits/test_split.csv",
}

for split_name, path in paths.items():
    df = pd.read_csv(path)
    print("\n" + "=" * 45)
    print(f"KÜME: {split_name}")
    print("=" * 45)
    print(f"Hasta Sayısı: {df['Patient ID'].nunique()}")
    print(f"Görüntü Sayısı: {len(df)}")
    print("-" * 20)

    for label in NIH_LABELS:
        count = df["Finding Labels"].fillna("").apply(
            lambda x: label in str(x).split("|")
        ).sum()
        print(f"{label:<20}: {count}")