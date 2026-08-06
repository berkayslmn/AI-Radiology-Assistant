import streamlit as st
import google.generativeai as genai


def generate_medical_report(findings_text):
    try:
        api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

        if not api_key:
            return "Hata: secrets.toml dosyasında geçerli bir API anahtarı bulunamadı."

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-3.6-flash")

        prompt = f"""
        Sen uzman bir radyologsun. Aşağıda bir derin öğrenme modelinin akciğer röntgeni üzerinden yaptığı tahminler verilmiştir. 
        Sadece bu bulguları kullanarak profesyonel, tıbbi bir ön değerlendirme raporu taslağı oluştur.

        Bulgular: {findings_text}

        Lütfen raporu aşağıdaki başlıklara göre yapılandır:
        - İNCELEME TÜRÜ: Posteroanterior (PA) Akciğer Radyografisi
        - BULGULAR: (Bulguları tıbbi bir dille açıkla)
        - SONUÇ: (Maddeler halinde özetle ve ileri tetkik/uzman değerlendirmesi öner)

        Önemli Not: Kesinlikle 'Bulgular' kısmında belirtilmeyen bir hastalığı veya anomalisi varmış gibi yorumlama. Sadece sana verilen bulgulara sadık kal.
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Rapor oluşturulurken bir hata oluştu: {str(e)}"