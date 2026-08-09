import pandas as pd

# Kendi CSV yolunu yazıyorsun
CSV_PATH = "veriseti/Filtered_Data_Entry.csv"
df = pd.read_csv(CSV_PATH)

toplam_gorsel = len(df)
essiz_hasta = df["Patient ID"].nunique()

print("=== VERİ SETİ ÖZETİ ===")
print(f"Toplam Görsel Sayısı: {toplam_gorsel}")
print(f"Eşsiz (Farklı) Hasta Sayısı: {essiz_hasta}")
print(f"Hasta Başına Ortalama Görsel: {toplam_gorsel / essiz_hasta:.2f}")
print("=======================\n")

print("En çok görsele sahip ilk 5 hasta (Rekortmenler):")
print(df["Patient ID"].value_counts().head(5))