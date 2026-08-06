import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import json
import os
from utils.dataset import NIH_LABELS
from utils.report_generator import generate_medical_report

# ==========================================
# 1. OPTİMİZE EDİLMİŞ EŞİK DEĞERLERİNİ YÜKLEME
# ==========================================
# Dinamik eşikleri az önce oluşturduğumuz JSON dosyasından otomatik okuyoruz
try:
    with open("models/best_thresholds.json", "r") as f:
        OPTIMAL_THRESHOLDS = json.load(f)
except FileNotFoundError:
    st.error("🚨 best_thresholds.json bulunamadı! Lütfen önce optimize_thresholds.py scriptini çalıştırın.")
    st.stop()

st.set_page_config(page_title="Karar Destek Sistemi", page_icon="🩺", layout="wide")


@st.cache_resource
def load_cnn_model():
    # Eski ResNet18 yerine yeni şampiyonumuz DenseNet121 mimarisine geçiş yapıldı
    cnn = models.densenet121(weights=None)

    # Eğitim aşamasındaki mimariyle tam uyumlu olması için Dropout katmanı eklendi
    # Not: DenseNet'te 'fc' değil, 'classifier' kullanılır
    num_ftrs = cnn.classifier.in_features
    cnn.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 14)
    )

    try:
        # Yeni eğitilen ve kalibre edilen DenseNet modelinin dosya yolu
        cnn.load_state_dict(
            torch.load("models/densenet121_kalibre_edilmis_model.pth", map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error("🚨 Eğitilmiş model bulunamadı! Lütfen modelin 'models' klasöründe olduğundan emin olun.")
        st.stop()

    cnn.eval()
    return cnn


cnn_model = load_cnn_model()

# Eğitim (train.py) ve doğrulama sürecindeki transform adımlarıyla BİREBİR AYNI olacak şekilde güncellendi
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ==========================================
# 2. DİNAMİK ETİKET ÇIKARMA FONKSİYONU
# ==========================================
def etiketleri_cikar(olasiliklar_tensoru):
    bulgular = []
    for i, olasilik in enumerate(olasiliklar_tensoru):
        hastalik_adi = NIH_LABELS[i]

        # O hastalığa özel JSON'dan okunan dinamik eşik değerini al (bulamazsa güvenli liman %50 kullan)
        esik_degeri = OPTIMAL_THRESHOLDS.get(hastalik_adi, 0.5)

        # Olasılık, o hastalığa özel optimize edilmiş eşiği aşıyorsa listeye ekle
        if olasilik.item() > esik_degeri:
            bulgular.append(f"{hastalik_adi} (%{int(olasilik.item() * 100)})")

    return bulgular if bulgular else ["Belirgin bir anomali tespit edilemedi (Normal)"]


# ==========================================
# 3. ARAYÜZ (UI) TASARIMI
# ==========================================
st.title("Yapay Zekâ Destekli Radyolojik Ön Değerlendirme Taslağı")
st.markdown("**Mimari:** PyTorch/DenseNet121 Çok Etiketli Sınıflandırma + Dinamik Eşik + Gemini LLM")
st.warning(
    "⚠️ **YASAL UYARI:** Bu çıktı otomatik oluşturulmuştur. Klinik tanı yerine geçmez, sadece karar destek amaçlıdır ve kesinlikle uzman değerlendirmesi gerektirir."
)
st.divider()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Görüntü Yükleme")
    uploaded_file = st.file_uploader("X-Ray görseli yükleyin", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Yüklenen X-Ray", use_container_width=True)

with col2:
    st.subheader("2. Otomatik Analiz ve Rapor")

    st.info(
        "🧠 **Akıllı Analiz:** Sistem, yanlış pozitifleri önlemek için her hastalığa özel dinamik karar eşik değerleri (F1-Score Optimizasyonu) kullanmaktadır."
    )

    if uploaded_file is not None:
        if st.button("🚀 Uçtan Uca Sistemi Çalıştır", use_container_width=True):
            with st.spinner("DenseNet121 tahminleri hesaplanıyor..."):
                input_tensor = preprocess(image).unsqueeze(0)

                with torch.no_grad():
                    outputs = cnn_model(input_tensor)
                    probabilities = torch.sigmoid(outputs[0])

                # Hata ayıklama paneli: Modelin ham olasılıkları ve JSON'dan gelen ideal eşikler kıyaslanıyor
                with st.expander("📊 Modelin Tüm Ham Olasılıklarını Gör"):
                    for i, olasilik in enumerate(probabilities):
                        hastalik = NIH_LABELS[i]
                        ideal_esik = OPTIMAL_THRESHOLDS.get(hastalik, 0.5) * 100
                        st.write(f"- **{hastalik}**: %{olasilik.item() * 100:.2f} *(Gerekli Eşik: %{ideal_esik:.2f})*")

                bulgu_metni = ", ".join(etiketleri_cikar(probabilities))

                st.success("✅ Derin Öğrenme Sınıflandırması Tamamlandı!")
                st.success(f"**Modelin Tahminleri:** {bulgu_metni}")

            with st.spinner("Gemini (LLM) profesyonel raporu oluşturuyor..."):
                rapor_metni = generate_medical_report(bulgu_metni)
                st.markdown("### 📝 LLM Ön Değerlendirme Taslağı")
                st.markdown(rapor_metni)