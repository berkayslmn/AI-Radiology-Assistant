# 🩺 AI-Powered Radiological Pre-Reporting Assistant

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B?style=flat-square&logo=streamlit)
![Gemini](https://img.shields.io/badge/Gemini-LLM%20Integration-8E75B2?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## 📌 Proje Hakkında ve Vizyon

Modern sağlık sistemlerinde radyoloji departmanları, artan tıbbi görüntüleme talepleri nedeniyle devasa bir iş yüküyle karşı karşıyadır. Bu proje, radyologların omuzundaki bu raporlama yükünü hafifletmek, görüntü analizi süreçlerini asiste ederek hızlandırmak ve gözden kaçabilecek anomalilerde erken teşhis doğruluğunu artırmak amacıyla tasarlanmış uçtan uca (end-to-end) bir **Yapay Zekâ Karar Destek Sistemidir.**

Geleneksel medikal yapay zekâ uygulamaları görüntüyü işleyip sadece sayısal bir anomali yüzdesi sunan "siyah kutu" (black-box) modelleriyken, bu sistem **Açıklanabilir Yapay Zekâ (Explainable AI - XAI)** prensiplerini merkeze alır. Model, tespit ettiği hastalıklar için karar verirken nereye odaklandığını ısı haritalarıyla kanıtlar ve ardından bu bulguları Üretken Yapay Zekâ (GenAI) modeline aktararak, uzman hekimlerin alışkın olduğu formatta tıbbi bir ön rapor taslağı oluşturur.

---

## ⚙️ Teknik Mimari ve Kullanılan Teknolojiler (Tech Stack)

Proje, ileri seviye bilgisayarlı görü yaklaşımları ile en modern büyük dil modellerini (LLM) aynı boru hattında birleştiren hibrit ve dinamik bir mimariye sahiptir:

*   **Derin Öğrenme ve Bilgisayarlı Görü (Computer Vision):** PyTorch kütüphanesi üzerinde çalışan **DenseNet121** mimarisi kullanılmaktadır. Model, Transfer Learning teknikleri ile eğitilmiş olup, 14 farklı torasik anomaliyi aynı anda tespit edebilecek **Çok Etiketli Sınıflandırma (Multi-Label Classification)** yapısına sahiptir.
*   **Açıklanabilir Yapay Zeka (XAI):** Siyah kutu problemini aşmak için **Grad-CAM (Gradient-weighted Class Activation Mapping)** algoritması entegre edilmiştir. Sistem, tespit edilen her bir hastalık için modelin evrişim (convolution) katmanlarındaki gradyanları hesaplar ve röntgen üzerinde milimetrik ısı haritaları (Heatmap) üretir.
*   **Dinamik Karar Eşikleri (Threshold Optimization):** Yanlış negatif ve yanlış pozitif dengesini sağlamak adına sabit bir %50 eşik değeri yerine, her sınıf için ROC-AUC analizleri ve **F1-Score maksimizasyonu** ile hesaplanmış özel dinamik eşik değerleri kullanılmaktadır.
*   **Üretken Yapay Zekâ ve Doğal Dil İşleme (GenAI & NLP):** Klinik bulguları anlamlı bir metne dönüştürmek için **Gemini LLM** kullanılmaktadır. İleri düzey Prompt Engineering teknikleri ile modelin halüsinasyon görmesi engellenmiş ve sadece CNN'den gelen kanıta dayalı, profesyonel bir *Posteroanterior (PA) Akciğer Grafisi Raporu* üretmesi sağlanmıştır.
*   **Veri Seti:** Dünya standartlarında kabul gören **NIH Chest X-ray14** veri setinin dikkatle dengelenmiş bir alt kümesi kullanılmıştır.
*   **Klinik Arayüz (UI):** Doktorların kodlama bilgisine ihtiyaç duymadan saniyeler içinde analiz alabileceği, medikal standartlara uygun **Streamlit** tabanlı interaktif bir arayüz geliştirilmiştir.

---

## 🔄 Sistem Akışı ve Metodoloji

Projenin çalışma mekanizması, birbirini besleyen ardışık beş ana aşamadan oluşur:

1.  **Görüntü Ön İşleme:** Sisteme yüklenen ham röntgen görüntüleri (224x224) yeniden boyutlandırılır ve ImageNet standartlarına göre normalize edilerek tensörlere dönüştürülür.
2.  **DenseNet121 Çıkarımı:** Normalize edilmiş görüntü evrişimli sinir ağından geçirilir. Sigmoid aktivasyon fonksiyonu ile 14 farklı hastalık için `0` ile `1` arasında bağımsız olasılık skorları (logits) hesaplanır.
3.  **Dinamik Bulgu Filtreleme:** Her bir hastalığın olasılık skoru, kalibrasyon aşamasında belirlenen kendine has optimum eşik değerinden (threshold) geçirilir. Yalnızca bu matematiksel barajı aşan bulgular sisteme dahil edilir.
4.  **Grad-CAM Görselleştirme:** Barajı aşan her bulgu için sistem geriye dönük bir analiz yapar. Modelin o teşhisi koyarken görüntünün hangi bölgesine (örn: sağ hiler bölge, bazal alanlar) odaklandığını kanıtlayan termal ısı haritaları orijinal röntgenin üzerine giydirilir.
5.  **LLM Ön Rapor Üretimi:** Ayıklanan bulgular "Uzman bir radyolog" personasıyla Gemini LLM'e iletilir. Model, "İnceleme Türü", "Bulgular" ve "Sonuç/Öneriler" başlıkları altında yapılandırılmış ön raporu kullanıcıya sunar.

---

## 👨‍⚕️ Mühendislik Vizyonu ve Sürekli Geliştirme (Human-in-the-Loop)

Sağlık teknolojilerinde mühendislik kadar klinik geçerlilik de hayati önem taşır. Bu proje, donanım, yazılım ve veri akışını bir bütün olarak ele alan sistem entegrasyonu disipliniyle yürütülmektedir.
*   **Geri Bildirim Döngüsü:** Modelin karar mekanizmaları ve ürettiği tıbbi terminolojinin tutarlılığı, uzman radyolog hekimlerin geri bildirimleriyle sürekli optimize edilmektedir.
*   **Risk ve Performans Optimizasyonu:** Erken durdurma (Early Stopping) mekanizmalarıyla aşırı öğrenme (overfitting) engellenmiş; klinik bir araca dönüşebilmesi için algoritmaların "Precision-Recall Trade-off" dengesi gerçek hayat senaryolarına göre kalibre edilmiştir.

---

## 🛡️ Güvenlik, Gizlilik ve Etik Standartlar

*   **Veri Mahremiyeti:** KVKK ve HIPAA regülasyonları gözetilerek, eğitim aşamasında kullanılan tüm veri setleri tamamen anonimleştirilmiş, açık kaynaklı akademik depolardan sağlanmıştır. Proje havuzunda veya sistem loglarında hiçbir gerçek hasta verisi veya kişisel sağlık bilgisi (PHI) barındırılmamaktadır.
*   **Yasal Uyarı (Disclaimer):** Bu yapay zekâ asistanı kesinlikle bir **tanı cihazı değildir**. Geliştirilen sistem, hekimin yerini almayı değil, hekimin karar alma sürecini asiste etmeyi amaçlayan bir **"Karar Destek Aracıdır"**. Üretilen tüm raporlar "taslak" statüsündedir ve nihai tanı her zaman uzman bir doktorun onayına tabidir.

---

> **Geliştirici:** Berkay Salman | Mekatronik Mühendisliği, İstanbul Ticaret Üniversitesi
> *Bilgisayarlı Görü, Üretken Yapay Zekâ ve Sistem Entegrasyonu Uygulamalı Mühendislik Projesi*
