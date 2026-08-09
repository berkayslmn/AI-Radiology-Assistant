import logging
import os

import google.generativeai as genai
import streamlit as st

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.6-flash"

NO_FINDING_TEXT = "Belirgin bir anomali tespit edilemedi (Normal)"

NO_FINDING_REPORT = """**İNCELEME TÜRÜ:** Posteroanterior (PA) Akciğer Radyografisi

**BULGULAR:** Değerlendirilen görüntüde, sistem tarafından kalibre edilmiş eşik değerlerini aşan bir anomali tespit edilmemiştir.

**SONUÇ:**
- Otomatik analizde belirgin bir patolojik bulguya rastlanmamıştır.
- Bu sonuç klinik muayenenin yerini tutmaz; semptom veya risk faktörü mevcutsa uzman radyolog değerlendirmesi önerilir.
"""

REPORT_PROMPT_TEMPLATE = """
Sen uzman bir radyologsun. Aşağıda bir derin öğrenme modelinin akciğer röntgeni üzerinden tespit ettiği bulgular ve model güven skorları verilmiştir.

Bulgular: {findings_text}

Bu bulguları kullanarak profesyonel, tıbbi bir ön değerlendirme raporu taslağı oluştur.

Kurallar:
1. Sadece verilen bulguları kullan.
2. Listede olmayan hastalık veya anomali ekleme.
3. Görüntüyü doğrudan görmediğin için lezyon boyutu, kesin lokalizasyon veya doku özellikleri hakkında varsayım yapma.
4. Kesin tanı koyma.
5. Hasta kimliği, yaş veya cinsiyet hakkında varsayım yapma.
6. Raporun sonunda çıktının yapay zekâ tarafından oluşturulmuş bir ön taslak olduğunu ve uzman hekim değerlendirmesi gerektirdiğini belirt.

Raporu şu başlıklarla oluştur:

- İNCELEME TÜRÜ: Posteroanterior (PA) Akciğer Radyografisi
- BULGULAR
- SONUÇ
"""


def _get_api_key():
    try:
        key = (
            st.secrets.get("GEMINI_API_KEY")
            or st.secrets.get("GOOGLE_API_KEY")
        )

        if key:
            return key

    except Exception:
        pass

    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )


def generate_medical_report(findings_text: str) -> str:

    if findings_text.strip() == NO_FINDING_TEXT:
        return NO_FINDING_REPORT

    api_key = _get_api_key()

    if not api_key:
        logger.error("Gemini API anahtarı bulunamadı.")

        return (
            "Hata: Geçerli bir API anahtarı bulunamadı. "
            "secrets.toml dosyanızı veya GEMINI_API_KEY "
            "ortam değişkeninizi kontrol edin."
        )

    try:
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(MODEL_NAME)

        prompt = REPORT_PROMPT_TEMPLATE.format(
            findings_text=findings_text
        )

        response = model.generate_content(prompt)

        if not getattr(response, "text", None):
            return "Hata: Model boş bir yanıt döndürdü."

        return response.text

    except Exception as exc:
        logger.exception(
            "Gemini rapor üretimi sırasında hata oluştu."
        )

        return (
            f"Rapor oluşturulurken bir hata oluştu: {exc}"
        )
