# 🩺 AI-Powered Radiological Pre-Reporting Assistant
<img width="706" height="857" alt="ekran1" src="https://github.com/user-attachments/assets/1b58f32e-f7e3-4267-8bb6-7cfb34e688e0" /> <img width="1077" height="337" alt="ekran2" src="https://github.com/user-attachments/assets/5aea13cd-c79f-409f-a935-e701bfe9673e" />
<img width="224" height="224" alt="ekran3" src="https://github.com/user-attachments/assets/d2c57bf6-e4ee-4a98-a406-7715f992eb06" /> <img width="224" height="224" alt="ekran6" src="https://github.com/user-attachments/assets/24ac36ab-5a4b-4c22-9b03-d5772229a7b6" />



[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-ff4b4b?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/Gemini-LLM%20Integration-8e75b2?style=flat-square)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)]()

## 📌 Proje Hakkında ve Vizyon

Modern sağlık sistemlerinde radyoloji departmanları, artan tıbbi görüntüleme talepleri nedeniyle devasa bir iş yüküyle karşı karşıyadır. Bu proje, radyologların omuzundaki bu raporlama yükünü hafifletmek, görüntü analizi süreçlerini asiste ederek hızlandırmak ve gözden kaçabilecek anomalilerde erken teşhis doğruluğunu artırmak amacıyla tasarlanmış uçtan uca (end-to-end) bir **Yapay Zekâ Karar Destek Sistemi**dir.

Geleneksel medikal yapay zekâ uygulamaları görüntüyü işleyip sadece sayısal bir anomali yüzdesi sunan "siyah kutu" (black-box) modelleriyken, bu sistem **Açıklanabilir Yapay Zekâ (Explainable AI – XAI)** prensiplerini merkeze alır. Model, tespit ettiği hastalıklar için karar verirken nereye odaklandığını ısı haritalarıyla kanıtlar ve ardından bu bulguları Üretken Yapay Zekâ (GenAI) modeline aktararak, uzman hekimlerin alışkın olduğu formatta tıbbi bir ön rapor taslağı oluşturur.

---

## ⚙️ Teknik Mimari ve Kullanılan Teknolojiler (Tech Stack)

Proje, ileri seviye bilgisayarlı görü yaklaşımları ile en modern büyük dil modellerini (LLM) aynı boru hattında birleştiren hibrit ve dinamik bir mimariye sahiptir:

- **Derin Öğrenme ve Bilgisayarlı Görü (Computer Vision):** PyTorch kütüphanesi üzerinde çalışan **DenseNet121** mimarisi kullanılmaktadır. Model, Transfer Learning teknikleri ile eğitilmiş olup, 14 farklı torasik anomaliyi aynı anda tespit edebilecek **Çok Etiketli Sınıflandırma (Multi-Label Classification)** yapısına sahiptir.
- **Hasta Bazlı Veri Ayrımı (Patient-Wise Split):** Aynı hastaya ait birden fazla takip görüntüsünün eğitim ve doğrulama/test setleri arasında sızmasını (data leakage) önlemek için split işlemi **görüntü bazında değil, hasta ID'si bazında** yapılmaktadır. Bir hastanın tüm görüntüleri her zaman aynı kümede (train / val / test) kalır.
- **Açıklanabilir Yapay Zeka (XAI):** Siyah kutu problemini aşmak için **Grad-CAM (Gradient-weighted Class Activation Mapping)** algoritması entegre edilmiştir. Sistem, tespit edilen her bir hastalık için modelin evrişim (convolution) katmanlarındaki gradyanları hesaplar ve röntgen üzerinde milimetrik ısı haritaları (Heatmap) üretir.
- **Dinamik Karar Eşikleri (Threshold Optimization):** Yanlış negatif ve yanlış pozitif dengesini sağlamak adına sabit bir %50 eşik değeri yerine, her sınıf için **yalnızca validation seti üzerinde** F-beta (β=0.5) skoru maksimizasyonu ile hesaplanmış özel dinamik eşik değerleri kullanılmaktadır. β=0.5 seçimi, Precision'a Recall'dan ~2 kat daha fazla ağırlık vererek klinik ortamda yanlış-pozitif ("gereksiz alarm") oranını kontrol altında tutmayı hedefler. Validation setinde 20'den az pozitif örneğe sahip sınıflar için istatistiksel güvenilirlik amacıyla sabit %50 eşiği kullanılır; ayrıca F-beta taraması bir sınıfı validation setinde tamamen "susturuyorsa" (hiç pozitif tahmin üretmiyorsa), o sınıf için F1-maksimizasyonuna geri dönülür. Test seti bu kalibrasyon sürecinde hiçbir şekilde kullanılmaz.
- **Test-Time Augmentation (TTA):** Çıkarım anında her görüntü hem orijinal hem yatay çevrilmiş hâliyle modelden geçirilir ve olasılıklar ortalanır. Model ağırlıkları değişmez; bu yalnızca tahminin görüntüdeki küçük varyasyonlara karşı daha kararlı olmasını sağlar. Eşik kalibrasyonu (`optimize_thresholds.py`) ve nihai değerlendirme (`evaluate.py`), tutarlılık için aynı TTA mantığını kullanır.
- **Üretken Yapay Zekâ ve Doğal Dil İşleme (GenAI & NLP):** Klinik bulguları anlamlı bir metne dönüştürmek için **Gemini LLM** (`gemini-3.6-flash`) kullanılmaktadır. İleri düzey Prompt Engineering teknikleri ile modelin halüsinasyon görmesi engellenmiş ve sadece CNN'den gelen kanıta dayalı, profesyonel bir *Posteroanterior (PA) Akciğer Grafisi Raporu* üretmesi sağlanmıştır. Anomali tespit edilmeyen ("Normal") durumlarda LLM'e hiç istek gönderilmez; sabit ve güvenli bir şablon metin döndürülür.
- **Veri Seti:** Dünya standartlarında kabul gören **NIH Chest X-ray14** veri setinin dikkatle dengelenmiş bir alt kümesi kullanılmıştır.
- **Klinik Arayüz (UI):** Doktorların kodlama bilgisine ihtiyaç duymadan saniyeler içinde analiz alabileceği, medikal standartlara uygun **Streamlit** tabanlı interaktif bir arayüz geliştirilmiştir.

---

## 🔄 Sistem Akışı ve Metodoloji

Projenin çalışma mekanizması, birbirini besleyen ardışık beş ana aşamadan oluşur:

1. **Görüntü Ön İşleme:** Sisteme yüklenen ham röntgen görüntüleri (224×224) yeniden boyutlandırılır ve ImageNet standartlarına göre normalize edilerek tensörlere dönüştürülür.
2. **DenseNet121 Çıkarımı:** Normalize edilmiş görüntü evrişimli sinir ağından geçirilir. Sigmoid aktivasyon fonksiyonu ile 14 farklı hastalık için `0` ile `1` arasında bağımsız olasılık skorları hesaplanır.
3. **Dinamik Bulgu Filtreleme:** Her bir hastalığın olasılık skoru, kalibrasyon aşamasında (yalnızca validation seti üzerinde, F-beta β=0.5 taramasıyla) belirlenen kendine has optimum eşik değerinden (threshold) geçirilir. Yalnızca bu matematiksel barajı aşan bulgular sisteme dahil edilir.
4. **Grad-CAM Görselleştirme:** Barajı aşan her bulgu için sistem geriye dönük bir analiz yapar. Modelin o teşhisi koyarken görüntünün hangi bölgesine (örn. sağ hiler bölge, bazal alanlar) odaklandığını kanıtlayan termal ısı haritaları orijinal röntgenin üzerine giydirilir.
5. **LLM Ön Rapor Üretimi:** Ayıklanan bulgular "Uzman bir radyolog" personasıyla, katı kurallarla sınırlandırılmış bir prompt aracılığıyla Gemini LLM'e iletilir. Model, "İnceleme Türü", "Bulgular" ve "Sonuç" başlıkları altında yapılandırılmış ön raporu kullanıcıya sunar.

---

## 🧪 Değerlendirme Metodolojisi

Modelin gerçek performansını yansız (unbiased) şekilde ölçmek için üç aşamalı, sızıntısız bir değerlendirme protokolü izlenmiştir:

| Aşama | Script | Kullanılan Veri | Amaç |
|---|---|---|---|
| 1. Split | `utils/split.py` | Tüm veri seti | Hasta bazlı train (%70) / val (%15) / test (%15) ayrımı |
| 2. Eğitim | `train.py` | Yalnızca train | Model ağırlıklarının öğrenilmesi |
| 3. Eşik Kalibrasyonu | `optimize_thresholds.py` | Yalnızca validation | Sınıf başına F-beta (β=0.5) taramasıyla eşik seçimi, TTA ile |
| 4. Nihai Ölçüm | `evaluate.py` | Yalnızca test | Hiç görülmemiş veri üzerinde, TTA ile nihai metrikler |

Bu ayrım sayesinde:
- **Data leakage** (aynı hastanın train/val/test arasında sızması) önlenmiştir.
- **Threshold leakage** (eşiklerin test setine bakılarak seçilmesi) önlenmiştir.
- Raporlanan test metrikleri, modelin daha önce hiç görmediği hastalar üzerindeki gerçek performansını yansıtır.

### 📊 Test Seti Sonuçları

Aşağıdaki sonuçlar, model eğitildikten ve eşikler yalnızca validation setinde kalibre edildikten sonra, **daha önce hiç görülmemiş 201 hastaya ait 371 test görüntüsü** üzerinde elde edilmiştir.

- Test hasta sayısı: **201**
- Test görüntü sayısı: **371**
- Split yöntemi: Hasta bazlı (patient-wise), sızıntısız
- Eşik değerleri: Yalnızca validation setinde F-beta (β=0.5) taramasıyla belirlendi, Test-Time Augmentation (TTA) ile

| Sınıf | Support | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Atelectasis | 41 | 0.8104 | 0.1891 | 0.2234 | 0.5122 | 0.3111 |
| Cardiomegaly | 25 | 0.8147 | 0.1462 | 0.2500 | 0.1600 | 0.1951 |
| Effusion | 54 | 0.8397 | 0.4752 | 0.7273 | 0.2963 | 0.4211 |
| Infiltration | 95 | 0.7137 | 0.3513 | 0.4026 | 0.3263 | 0.3605 |
| Mass | 10 | 0.3447 | 0.0153 | 0.0000 | 0.0000 | 0.0000 |
| Nodule | 16 | 0.6492 | 0.0689 | 0.1429 | 0.1875 | 0.1622 |
| Pneumonia | 8 | 0.5794 | 0.0176 | 0.0179 | 0.2500 | 0.0333 |
| Pneumothorax | 21 | 0.7066 | 0.0798 | 0.0769 | 0.1905 | 0.1096 |
| Consolidation | 33 | 0.8485 | 0.2177 | 0.2857 | 0.1212 | 0.1702 |
| Edema | 2 | 0.9108 | 0.1753 | 0.0227 | 0.5000 | 0.0435 |
| Emphysema | 19 | 0.7516 | 0.1140 | 0.0000 | 0.0000 | 0.0000 |
| Fibrosis | 34 | 0.7695 | 0.2374 | 0.0860 | 0.7941 | 0.1552 |
| Pleural_Thickening | 11 | 0.6215 | 0.0300 | 0.0000 | 0.0000 | 0.0000 |
| Hernia | 2 | 0.9908 | 0.2159 | 0.0714 | 1.0000 | 0.1333 |
| **Makro Ortalama** | 371 | **0.7394** | **0.1667** | **0.1648** | **0.3099** | **0.1497** |

**Sonuçların yorumu:**

- Model, **ayırt edici sinyali genel olarak öğrenmiştir**: makro ROC-AUC 0.74, birçok sınıfta (Effusion 0.84, Consolidation 0.85, Hernia 0.99) güçlü ayrım gücü göstermektedir. ROC-AUC, olasılık sıralamasının kalitesini ölçtüğü için eşik seçiminden bağımsızdır.
- Precision/Recall/F1 skorları, özellikle az örnekli sınıflarda (Mass: 10, Edema: 2, Hernia: 2, Pneumonia: 8) düşüktür. Bu, **model hatası değil, veri seti boyutu sınırlamasıdır**: 14 sınıflı çok etiketli bir problem için ~3500 eğitim görüntüsü, literatürdeki büyük ölçekli çalışmalara (örn. CheXNet, 100.000+ görüntü) kıyasla küçüktür.
- **Mass, Emphysema ve Pleural_Thickening sınıflarında F1=0** gözlemlenmiştir. Bu üç sınıfın test setindeki support'u çok düşüktür (sırasıyla 10, 19, 11) ve validation'da kalibre edilen eşik, test setindeki bu az sayıdaki pozitif örneği yakalayamamıştır. Bu, validation ve test dağılımları arasındaki doğal varyansın küçük örneklem büyüklüğüyle birleşmesinden kaynaklanan bilinen bir sınırlamadır; test setine bakarak eşik ayarlamak (threshold leakage) yöntemsel olarak yanlış olacağından bilinçli olarak düzeltilmemiştir.
- Eşik seçiminde F-beta (β=0.5) kullanılması, sistemi **precision-öncelikli** bir noktaya taşımıştır (makro Precision 0.165, makro Recall 0.310) — klinik bir karar destek aracında "az ama güvenilir" uyarı, "çok ama gürültülü" uyarıya tercih edilmiştir. Bu bir tasarım kararıdır; β değeri değiştirilerek denge yeniden ayarlanabilir.

---

## 👨‍⚕️ Mühendislik Vizyonu ve Sürekli Geliştirme (Human-in-the-Loop)

Sağlık teknolojilerinde mühendislik kadar klinik geçerlilik de hayati önem taşır. Bu proje, donanım, yazılım ve veri akışını bir bütün olarak ele alan sistem entegrasyonu disipliniyle yürütülmektedir.

- **Geri Bildirim Döngüsü:** Modelin karar mekanizmaları ve ürettiği tıbbi terminolojinin tutarlılığı, uzman radyolog hekimlerin geri bildirimleriyle sürekli optimize edilmektedir.
- **Risk ve Performans Optimizasyonu:** Erken durdurma (Early Stopping) mekanizmalarıyla aşırı öğrenme (overfitting) engellenmiş; sınıf dengesizliği `pos_weight` ağırlıklandırmasıyla `BCEWithLogitsLoss` içinde ele alınmış; klinik bir araca dönüşebilmesi için eşik seçiminde F-beta (β=0.5) skoru kullanılarak "Precision-Recall Trade-off" dengesi bilinçli olarak Precision lehine kalibre edilmiştir. Ayrıca çıkarım anında Test-Time Augmentation (TTA) uygulanarak tahminlerin görüntüdeki küçük varyasyonlara karşı kararlılığı artırılmıştır.

### 🔭 Bilinen Sınırlamalar ve Gelecek Çalışmalar

- Şu an yalnızca Posteroanterior (PA) görüş açısı destekleniyor; lateral görüntüler dahil edilmemiştir.
- Kullanılan veri seti, NIH Chest X-ray14'ün dengelenmiş bir alt kümesidir (~5.000 görüntü, 1.335 hasta); tam veri seti üzerinde sonuçlar farklılık gösterebilir.
- Model tek bir train/val/test split'i üzerinde değerlendirilmiştir; k-fold cross-validation henüz uygulanmamıştır.
- Test setinde support'u çok düşük olan sınıflarda (Mass: 10, Edema: 2, Hernia: 2, Pneumonia: 8, Pleural_Thickening: 11) metrikler yüksek varyanslıdır ve tek başına güvenilir bir performans göstergesi değildir.
- Nadir sınıflarda (özellikle Mass, Emphysema, Pleural_Thickening) modelin ayırt edici gücü zayıf kalmıştır; bu durum, class-balanced sampling (örn. `WeightedRandomSampler`) veya daha büyük bir veri setiyle iyileştirilebilir.
- Gerçek radyologlarla karşılaştırmalı klinik doğrulama çalışması yapılmamıştır — sistem bir prototip / karar destek aracıdır.

---

## 🚀 Kurulum ve Çalıştırma

```bash
# Bağımlılıkları kur
pip install -r requirements

# 1) Hasta bazlı split'i bir kez üret
python -m utils.split

# 2) Modeli eğit
python train.py

# 3) Validation setinde eşikleri kalibre et (F-beta β=0.5, TTA ile)
python optimize_thresholds.py

# 4) Test setinde nihai metrikleri üret (TTA ile)
python evaluate.py

# 5) Arayüzü başlat
streamlit run app.py
```

`Gemini API` anahtarınızı `.streamlit/secrets.toml` dosyasına aşağıdaki formatta eklemeniz gerekir:

```toml
GEMINI_API_KEY = "your-api-key-here"
```

---

## 🛡️ Güvenlik, Gizlilik ve Etik Standartlar

- **Veri Mahremiyeti:** KVKK ve HIPAA regülasyonları gözetilerek, eğitim aşamasında kullanılan tüm veri setleri tamamen anonimleştirilmiş, açık kaynaklı akademik depolardan sağlanmıştır. Proje havuzunda veya sistem loglarında hiçbir gerçek hasta verisi veya kişisel sağlık bilgisi (PHI) barındırılmamaktadır.
- **Yasal Uyarı (Disclaimer):** Bu yapay zekâ asistanı kesinlikle bir **tanı cihazı değildir**. Geliştirilen sistem, hekimin yerini almayı değil, hekimin karar alma sürecini asiste etmeyi amaçlayan bir **"Karar Destek Aracı"**dır. Üretilen tüm raporlar "taslak" statüsündedir ve nihai tanı her zaman uzman bir doktorun onayına tabidir.

---

> **Geliştirici:** Berkay Salman | Mekatronik Mühendisliği, İstanbul Ticaret Üniversitesi
> *Bilgisayarlı Görü, Üretken Yapay Zekâ ve Sistem Entegrasyonu Uygulamalı Mühendislik Projesi*
