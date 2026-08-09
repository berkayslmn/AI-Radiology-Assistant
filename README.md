# 🩺 AI Radiology Assistant (Uçtan Uca Karar Destek Sistemi)

Bu proje, göğüs hastalıklarının tespiti için derin öğrenme (Deep Learning), açıklanabilir yapay zeka (XAI - Grad-CAM) ve üretken yapay zekayı (Generative AI - LLM) tek bir boru hattında (pipeline) birleştiren kapsamlı bir medikal karar destek sistemidir. 

Sistem, NIH Chest X-ray14 veri seti üzerinde eğitilmiş olup, radyolojik bulguları tespit eder, ısı haritaları (heatmap) ile lezyon bölgelerini kanıtlar ve Gemini tabanlı doğal dil işleme modeli ile profesyonel bir tıbbi "Ön Değerlendirme Raporu" taslağı oluşturur.

## 🚀 Proje Mimarisi ve Mühendislik Yaklaşımları

Bu proje standart bir sınıflandırma ödevinin ötesinde, tıbbi yapay zeka geliştirme süreçlerindeki regülasyonlar ve endüstri standartları göz önüne alınarak tasarlanmıştır:

* **Patient-Wise Split (Veri Sızıntısı Koruması):** Aynı hastaya ait farklı röntgenlerin hem eğitim hem test setine düşmesi (Data Leakage) engellenmiş; hastalar %70 Eğitim, %15 Doğrulama ve %15 Test olacak şekilde izole edilmiştir.
* **Transfer Learning & Custom Head:** `Torchvision` üzerinden `DenseNet121` omurgası kullanılmış, sınıflandırıcı katman tıbbi veri setine özel olarak (Dropout ve BCEWithLogitsLoss ile) modifiye edilmiştir.
* **Dinamik Eşik Optimizasyonu (Threshold Calibration):** Model, standart %50 barajı yerine, her hastalık sınıfı için (Doğrulama seti üzerinden F1 skoru maksimizasyonu ile) dinamik eşikler hesaplayarak Yanlış Pozitif/Negatif (FP/FN) oranını optimize eder.
* **Erken Durdurma (Early Stopping) & LR Scheduler:** Aşırı öğrenmeyi (Overfitting) engellemek için validasyon kaybı izlenmiş ve dinamik öğrenme oranı planlaması yapılmıştır.
* **XAI (Grad-CAM):** Doktorlara modelin neden o teşhisi koyduğunu açıklamak adına, teşhisi etkileyen pikseller ısı haritası olarak (Weakly-supervised localization) sunulmuştur.
* **Maliyet ve Latency Optimizasyonu:** Normal/Sağlıklı tespit edilen röntgenler için LLM API'sine istek atılmayarak gecikme süresi (latency) ve maliyet düşürülmüş, halüsinasyon riski sıfırlanmıştır.

## 📊 Test Seti Performansı (ROC-AUC Skorları)

Modelin, eğitim sırasında **daha önce hiç görmediği** izole test seti (`test_split.csv`) üzerindeki ROC-AUC başarısı aşağıdadır:

| Hastalık Sınıfı | ROC-AUC Skoru | Optimum Karar Eşiği |
| :--- | :---: | :---: |
| **Hernia (Fıtık)** | **0.9905** | % 96.20 |
| **Edema (Ödem)** | **0.9644** | % 90.87 |
| **Cardiomegaly** | **0.8896** | % 78.31 |
| **Pneumothorax** | **0.8605** | % 83.80 |
| **Mass (Kitle)** | **0.8513** | % 77.37 |
| **Consolidation** | **0.8486** | % 78.64 |
| **Pleural Thickening** | **0.8436** | % 79.16 |
| **Effusion** | **0.8388** | % 58.24 |
| **Pneumonia** | **0.8369** | % 84.44 |
| **Fibrosis** | **0.8288** | % 80.20 |
| **Emphysema** | **0.8227** | % 78.80 |
| **Nodule** | **0.7821** | % 76.36 |
| **Atelectasis** | **0.7820** | % 59.31 |
| **Infiltration** | **0.6770** | % 51.63 |

## 🛠️ Kullanılan Teknolojiler (Tech Stack)

* **Deep Learning Framework:** PyTorch, Torchvision
* **XAI (Açıklanabilir YZ):** pytorch-grad-cam
* **GenAI (LLM):** Google Gemini Pro / Flash API
* **Veri Manipülasyonu:** Pandas, NumPy, Scikit-learn
* **Web & UI:** Streamlit

## ⚠️ Yasal Uyarı
*Bu sistem klinik tanı koymak için değil, doktorların iş akışını hızlandırmak ve ikinci bir görüş (second opinion) sunmak amacıyla tasarlanmış bir ön değerlendirme (triage) aracıdır. Kesin teşhis her zaman uzman hekimler tarafından konulmalıdır.*
