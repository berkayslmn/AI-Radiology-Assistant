import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import json
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils.dataset import NIH_LABELS
from utils.report_generator import generate_medical_report
 
try:
    with open("models/best_thresholds.json", "r") as f:
        OPTIMAL_THRESHOLDS = json.load(f)
except FileNotFoundError:
    st.error("🚨 best_thresholds.json bulunamadı! Lütfen önce optimize_thresholds.py scriptini çalıştırın.")
    st.stop()

st.set_page_config(page_title="Karar Destek Sistemi", page_icon="🩺", layout="wide")


@st.cache_resource
def load_cnn_model():
    cnn = models.densenet121(weights=None)
    num_ftrs = cnn.classifier.in_features
    cnn.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 14)
    )

    try:
        cnn.load_state_dict(
            torch.load("models/densenet121_kalibre_edilmis_model.pth", map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error("🚨 Eğitilmiş model bulunamadı! Lütfen modelin 'models' klasöründe olduğundan emin olun.")
        st.stop()

    cnn.eval()
    return cnn


cnn_model = load_cnn_model()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_predictions(probabilities):
    detected_labels = []
    detected_indices = []

    for i, olasilik in enumerate(probabilities):
        hastalik_adi = NIH_LABELS[i]
        esik_degeri = OPTIMAL_THRESHOLDS.get(hastalik_adi, 0.5)

        if olasilik.item() > esik_degeri:
            detected_labels.append(f"{hastalik_adi} (%{int(olasilik.item() * 100)})")
            detected_indices.append(i)

    return detected_labels, detected_indices


st.title("Yapay Zekâ Destekli Radyolojik Ön Değerlendirme Taslağı")
st.markdown("**Mimari:** PyTorch/DenseNet121 + Dinamik Eşik + **Grad-CAM (XAI)** + Gemini LLM")
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
        st.image(image, caption="Orijinal X-Ray", use_container_width=True)

with col2:
    st.subheader("2. Otomatik Analiz ve Rapor")
    st.info(
        "🧠 **Açıklanabilir Yapay Zeka (XAI):** Sistem, bulguları tespit ettikten sonra karar verirken odaklandığı pikselleri Grad-CAM algoritması ile ısı haritası (Heatmap) olarak işaretler."
    )

    if uploaded_file is not None:
        if st.button("🚀 Uçtan Uca Sistemi Çalıştır", use_container_width=True):

            with st.spinner("DenseNet121 teşhisleri hesaplanıyor..."):
                input_tensor = preprocess(image).unsqueeze(0)

                with torch.no_grad():
                    outputs = cnn_model(input_tensor)
                    probabilities = torch.sigmoid(outputs[0])

                detected_labels, detected_indices = get_predictions(probabilities)

                bulgu_metni = ", ".join(
                    detected_labels) if detected_labels else "Belirgin bir anomali tespit edilemedi (Normal)"

                with st.expander("📊 Modelin Tüm Ham Olasılıklarını Gör"):
                    for i, olasilik in enumerate(probabilities):
                        hastalik = NIH_LABELS[i]
                        ideal_esik = OPTIMAL_THRESHOLDS.get(hastalik, 0.5) * 100
                        st.write(f"- **{hastalik}**: %{olasilik.item() * 100:.2f} *(Gerekli Eşik: %{ideal_esik:.2f})*")

                st.success(f"**Modelin Tahminleri:** {bulgu_metni}")

                if detected_indices:
                    with st.spinner("Odak noktaları (Heatmap) oluşturuluyor..."):
                        target_layers = [cnn_model.features[-1]]

                        rgb_img = np.float32(image.resize((224, 224))) / 255

                        st.markdown("### 🔍 Açıklanabilir Yapay Zeka (Grad-CAM Analizi)")

                        for idx in detected_indices:
                            hastalik_adi = NIH_LABELS[idx]
                            targets = [ClassifierOutputTarget(idx)]

                            with GradCAM(model=cnn_model, target_layers=target_layers) as cam:
                                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

                                heatmap_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                                st.markdown("---")
                                st.markdown(f"#### {hastalik_adi}")

                                caption_text = f"**Grad-CAM Açıklaması:** Modelin \"{hastalik_adi}\" kararını verirken en fazla odaklandığı bölgeler kırmızı renk ile gösterilmektedir."

                                st.image(heatmap_image, caption=caption_text, use_container_width=True)

            if 'bulgu_metni' in locals():
                with st.spinner("Gemini (LLM) profesyonel raporu oluşturuyor..."):
                    rapor_metni = generate_medical_report(bulgu_metni)
                    st.markdown("### 📝 LLM Ön Değerlendirme Taslağı")
                    st.markdown(rapor_metni)
