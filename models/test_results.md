## Test Seti Sonuçları

- Test hasta sayısı: **201**
- Test görüntü sayısı: **652**
- Split yöntemi: Hasta bazlı (patient-wise), sızıntısız
- Eşik değerleri: Yalnızca validation setinde F-beta(0.5) taramasıyla belirlendi

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
| **Makro Ortalama** | 371 | 0.7394 | 0.1667 | 0.1648 | 0.3099 | 0.1497 |
