import pandas as pd
import os

image_dir = 'veriseti/images/'
original_csv = 'veriseti/Data_Entry_2017.csv'
filtered_csv = 'veriseti/Filtered_Data_Entry.csv'

print("Klasördeki görseller taranıyor...")
mevcut_gorseller = set(os.listdir(image_dir)) 

print("Orijinal CSV okunuyor...")
df = pd.read_csv(original_csv)

df_filtered = df[df['Image Index'].isin(mevcut_gorseller)]
df_filtered.to_csv(filtered_csv, index=False)

print(f"İşlem tamam! Orijinal satır sayısı: {len(df)}")
print(f"Filtrelenmiş satır sayısı (Eğitime girecek): {len(df_filtered)}")
print(f"Yeni dosya '{filtered_csv}' olarak kaydedildi.")
