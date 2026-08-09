# 🩺 AI Radiology Assistant (Uçtan Uca Karar Destek Sistemi)

Bu proje, göğüs hastalıklarının tespiti için derin öğrenme (Deep Learning), açıklanabilir yapay zeka (XAI - Grad-CAM) ve üretken yapay zekayı (Generative AI - LLM) tek bir boru hattında (pipeline) birleştiren kapsamlı bir medikal karar destek sistemidir. 

Sistem, NIH Chest X-ray14 veri seti üzerinde eğitilmiş olup, radyolojik bulguları tespit eder, ısı haritaları (heatmap) ile lezyon bölgelerini kanıtlar ve Gemini tabanlı doğal dil işleme modeli ile profesyonel bir tıbbi "Ön Değerlendirme Raporu" taslağı oluşturur.

## 🚀 Proje Mimarisi ve Mühendislik Yaklaşımları

Bu proje standart bir sınıflandırma ödevinin ötesinde, tıbbi yapay zeka geliştirme süreçlerindeki regülasyonlar ve endüstri standartları göz önüne alınarak tasarlanmıştır:

* **Patient-Wise Split (Veri Sızıntısı Koruması):** Aynı hastaya ait farklı röntgenlerin hem eğitim hem test setine düşmesi (Data Leakage) engellenmiş; hastalar %70 Eğitim, %15 Doğrulama ve %15 Test olacak şekilde izole edilmiştir.
* **Transfer Learning & Custom Head:** `Torchvision` üzerinden `DenseNet121` omurgası kullanılmış, sınıflandırıcı katman tıbbi veri setine özel olarak (Dropout ve BCEWithLogitsLoss ile) modifiye edilmiştir.
* **Dinamik Eşik Optimizasyonu (Threshold Calibration & Fallback):** F1 skorunun az örnekli sınıflardaki oynaklığını ve aşırı uçlara saplanma riskini önlemek için validasyon seti üzerinde dinamik tarama yapılmıştır. Düşük destekli (support < 20) sınıflarda ise istatistiksel güvenilirlik için sabit `0.5` güvenlik ağı (fallback) devreye sokulmuştur.
* **Erken Durdurma (Early Stopping) & LR Scheduler:** Aşırı öğrenmeyi (Overfitting) engellemek için validasyon kaybı izlenmiş ve dinamik öğrenme oranı planlaması yapılmıştır.
* **XAI (Grad-CAM):** Doktorlara modelin neden o teşhisi koyduğunu açıklamak adına, teşhisi etkileyen pikseller ısı haritası olarak (Weakly-supervised localization) sunulmuştur.
* **Maliyet ve Latency Optimizasyonu:** Normal/Sağlıklı tespit edilen röntgenler için LLM API'sine istek atılmayarak gecikme süresi (latency) ve maliyet düşürülmüş, halüsinasyon riski sıfırlanmıştır.

## 📊 Test Seti Performansı (Nihai Sonuçlar)

Modelin, eğitim ve eşik optimizasyonu süreçlerinde **daha önce hiç görmediği** izole test seti (`test_split.csv`) üzerindeki güncel performans metriği aşağıdadır:

| Sınıf | Support | ROC-AUC | PR-AUC | Precision | Recall | F1 Skoru |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Atelectasis** | 41 | 0.8104 | 0.1891 | 0.2234 | 0.5122 | 0.3111 |
| **Cardiomegaly** | 25 | 0.8147 | 0.1462 | 0.2500 | 0.1600 | 0.1951 |
| **Effusion** | 54 | 0.8397 | 0.4752 | 0.7273 | 0.2963 | 0.4211 |
| **Infiltration** | 95 | 0.7137 | 0.3513 | 0.4026 | 0.3263 | 0.3605 |
| **Mass** | 10 | 0.3447 | 0.0153 | 0.0000 | 0.0000 | 0.0000 |
| **Nodule** | 16 | 0.6492 | 0.0689 | 0.1429 | 0.1875 | 0.1622 |
| **Pneumonia** | 8 | 0.5794 | 0.0176 | 0.0179 | 0.2500 | 0.0333 |
| **Pneumothorax** | 21 | 0.7066 | 0.0798 | 0.0769 | 0.1905 | 0.1096 |
| **Consolidation** | 33 | 0.8485 | 0.2177 | 0.2857 | 0.1212 | 0.1702 |
| **Edema** | 2 | 0.9108 | 0.1753 | 0.0227 | 0.5000 | 0.0435 |
| **Emphysema** | 19 | 0.7516 | 0.1140 | 0.0000 | 0.0000 | 0.0000 |
| **Fibrosis** | 34 | 0.7695 | 0.2374 | 0.0860 | 0.7941 | 0.1552 |
| **Pleural Thickening** | 11 | 0.6215 | 0.0300 | 0.0000 | 0.0000 | 0.0000 |
| **Hernia** | 2 | 0.9908 | 0.2159 | 0.0714 | 1.0000 | 0.1333 |
| **MAKRO ORTALAMA** | **371** | **0.7394** | **0.1667** | **0.1648** | **0.3099** | **0.1497** |

## 🛠️ Kullanılan Teknolojiler (Tech Stack)

* **Deep Learning Framework:** PyTorch, Torchvision
* **XAI (Açıklanabilir YZ):** pytorch-grad-cam
* **GenAI (LLM):** Google Gemini Pro / Flash API
* **Veri Manipülasyonu:** Pandas, NumPy, Scikit-learn
* **Web & UI:** Streamlit

## ⚠️ Yasal Uyarı
*Bu sistem klinik tanı koymak için değil, doktorların iş akışını hızlandırmak ve ikinci bir görüş (second opinion) sunmak amacıyla tasarlanmış bir ön değerlendirme (triage) aracıdır. Kesin teşhis her zaman uzman hekimler tarafından konulmalıdır.*

* # 🧠 Sistem Mimarisi

```text
Chest X-Ray
    │
    ▼
Image Preprocessing
Resize / Normalize
    │
    ▼
DenseNet121
Transfer Learning
    │
    ▼
14-Class Multi-Label Logits
    │
    ▼
Sigmoid Scores
    │
    ▼
Class-Specific Decision Thresholds
Youden's J
    │
    ├───────────────┐
    ▼               ▼
Predictions      Grad-CAM
    │               │
    └───────┬───────┘
            ▼
        Gemini LLM
            │
            ▼
Preliminary Report Draft
            │
            ▼
      Streamlit UI

