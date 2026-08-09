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

Bulgular (SADECE bunlar, başka hiçbir şey değil): {findings_text}

GÖREV: Bu bulguları kullanarak profesyonel, tıbbi bir ön değerlendirme raporu taslağı oluştur.

KESİN KURALLAR (bunları ihlal etme):
1. SADECE yukarıda verilen bulgu listesini kullan. Listede olmayan hiçbir hastalık, anomali veya bulgudan bahsetme.
2. Görüntüyü kendin "görmüyorsun" -- yalnızca sana verilen metin bulgularını yorumluyorsun. Görmediğin detaylar (lezyon boyutu, kesin lokalizasyon, doku dansitesi vb.) hakkında spekülasyon yapma.
3. Kesin tanı koyma. Bu bir "ön değerlendirme taslağıdır", nihai tanı değildir.
4. Hasta kimliği, yaş, cinsiyet gibi bilgiler sana verilmedi; bunlar hakkında varsayımda bulunma.
5. Raporun sonunda mutlaka bu çıktının bir yapay zeka ön taslağı olduğunu ve uzman hekim onayı gerektirdiğini belirt.

Raporu şu başlıklarla yapılandır:
- İNCELEME TÜRÜ: Posteroanterior (PA) Akciğer Radyografisi
- BULGULAR: (Yalnızca verilen bulguları tıbbi bir dille açıkla)
- SONUÇ: (Maddeler halinde özetle ve ileri tetkik/uzman değerlendirmesi öner)
"""


def _get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        if key:
            return key
    except Exception:
        pass

    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def generate_medical_report(findings_text: str) -> str:
    if findings_text.strip() == NO_FINDING_TEXT:
        return NO_FINDING_REPORT

    api_key = _get_api_key()
    if not api_key:
        logger.error("Gemini API anahtarı bulunamadı.")
        return (
            "Hata: Geçerli bir API anahtarı bulunamadı. "
            "secrets.toml dosyanızı veya GEMINI_API_KEY ortam değişkeninizi kontrol edin."
        )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)

        prompt = REPORT_PROMPT_TEMPLATE.format(findings_text=findings_text)
        response = model.generate_content(prompt)

        if not getattr(response, "text", None):
            logger.warning(
                "Gemini boş yanıt döndürdü. prompt_feedback=%s",
                getattr(response, "prompt_feedback", "bilinmiyor"),
            )
            return "Hata: Model boş bir yanıt döndürdü (içerik güvenlik filtresine takılmış olabilir)."

        return response.text

    except Exception as exc:
        logger.exception("Gemini rapor üretimi sırasında hata oluştu.")
        return f"Rapor oluşturulurken bir hata oluştu: {exc}"