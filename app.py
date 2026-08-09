import json

import numpy as np
import streamlit as st
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from utils.dataset import NIH_LABELS
from utils.model import load_trained_densenet121, predict_with_tta
from utils.report_generator import generate_medical_report, NO_FINDING_TEXT

DEFAULT_THRESHOLD = 0.5
THRESHOLDS_PATH = "models/best_thresholds.json"
MODEL_PATH = "models/densenet121_kalibre_edilmis_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_thresholds():
    try:
        with open(THRESHOLDS_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"🚨 {THRESHOLDS_PATH} bulunamadı!")
        st.stop()


@st.cache_resource
def load_cnn_model():
    try:
        return load_trained_densenet121(MODEL_PATH, DEVICE)
    except FileNotFoundError:
        st.error(f"🚨 Eğitilmiş model bulunamadı! ('{MODEL_PATH}')")
        st.stop()


def get_threshold(disease_name: str) -> float:
    return OPTIMAL_THRESHOLDS.get(disease_name, DEFAULT_THRESHOLD)


def get_predictions(probabilities: torch.Tensor):
    detected_labels = []
    detected_indices = []
    for i, olasilik in enumerate(probabilities):
        hastalik_adi = NIH_LABELS[i]
        esik_degeri = get_threshold(hastalik_adi)
        if olasilik.item() > esik_degeri:
            detected_labels.append(f"{hastalik_adi} (%{int(olasilik.item() * 100)})")
            detected_indices.append(i)
    return detected_labels, detected_indices


OPTIMAL_THRESHOLDS = load_thresholds()

st.set_page_config(page_title="Karar Destek Sistemi", page_icon="🩺", layout="wide")

cnn_model = load_cnn_model()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.title("Yapay Zekâ Destekli Radyolojik Ön Değerlendirme Taslağı")
st.markdown("**Mimari:** PyTorch/DenseNet121 + Dinamik Eşik + **Grad-CAM (XAI)** + Gemini LLM")
st.warning(
    "⚠️ **YASAL UYARI:** Bu çıktı otomatik oluşturulmuştur. Klinik tanı yerine geçmez, "
    "sadece karar destek amaçlıdır ve kesinlikle uzman değerlendirmesi gerektirir."
)
st.divider()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. Görüntü Yükleme")
    uploaded_file = st.file_uploader("X-Ray görseli yükleyin", type=["jpg", "jpeg", "png"])
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Orijinal X-Ray", use_container_width=True)

with col2:
    st.subheader("2. Otomatik Analiz ve Rapor")
    st.info(
        "🧠 **Açıklanabilir Yapay Zeka (XAI):** Sistem, bulguları tespit ettikten sonra "
        "karar verirken odaklandığı pikselleri Grad-CAM algoritması ile ısı haritası "
        "(Heatmap) olarak işaretler."
    )

    if image is not None and st.button("🚀 Uçtan Uca Sistemi Çalıştır", use_container_width=True):
        with st.spinner("DenseNet121 teşhisleri hesaplanıyor (TTA ile)..."):
            input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
            probabilities = predict_with_tta(cnn_model, input_tensor)

            detected_labels, detected_indices = get_predictions(probabilities)
            bulgu_metni = (
                ", ".join(detected_labels)
                if detected_labels
                else NO_FINDING_TEXT
            )

            with st.expander("📊 Modelin Tüm Ham Olasılıklarını Gör"):
                for i, olasilik in enumerate(probabilities):
                    hastalik = NIH_LABELS[i]
                    ideal_esik = get_threshold(hastalik) * 100
                    st.write(
                        f"- **{hastalik}**: %{olasilik.item() * 100:.2f} "
                        f"*(Gerekli Eşik: %{ideal_esik:.2f})*"
                    )

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
                        caption_text = (
                            f"**Grad-CAM Açıklaması:** Modelin \"{hastalik_adi}\" kararını "
                            "verirken en fazla odaklandığı bölgeler kırmızı renk ile "
                            "gösterilmektedir."
                        )
                        st.image(heatmap_image, caption=caption_text, use_container_width=True)

        with st.spinner("Gemini (LLM) profesyonel raporu oluşturuyor..."):
            rapor_metni = generate_medical_report(bulgu_metni)
            st.markdown("### 📝 LLM Ön Değerlendirme Taslağı")
            st.markdown(rapor_metni)

    elif image is None:
        st.caption("Analizi başlatmak için önce soldan bir X-Ray görseli yükleyin.")