# IF3170_G35_Pistar

Tubes 2 IF3170 Inteligensi Artifisial: prediksi pembatalan booking hotel (`is_canceled`) pakai KNN, Decision Tree, Logistic Regression, SVM, dan Naive Bayes 

## Setup

```bash
pip install -r requirements.txt
```

Taruh `train.csv`, `test.csv`, `sample_submission.csv` dari Kaggle di folder `data/` (nggak di-commit).

## Run

Buka `notebooks/main.ipynb`, Run All. Hasilnya ada di `output/submission.csv`.

## Model Pilihan Dan Hasil

Model final adalah **KNN from scratch** dengan `k=11`, jarak **Manhattan**, bobot **invers jarak**, dan threshold `0.49`. Model menggunakan fitur dasar dan tanggal, target encoding, serta bobot mutual information (MI). KNN dipilih karena memperoleh mean **Macro F1 tertinggi pada konfirmasi 5-fold CV development, yaitu 0.8383**, di antara kandidat yang diuji. Threshold dipilih dari prediksi out-of-fold (OOF) development.

Skor Kaggle yang diperoleh adalah **0.84092**.

## Pembagian tugas

| Nama | NIM | Tugas |
|------|-----|-------|
| Ariel Cornelius Sitorus | 13524085 | Naive Bayes, graf kelas-fitur, serta integrasi model, bonus, dan notebook. |
| Rava Khoman Tuah Saragih | 13524107 | Decision Tree C4.5, penanganan kategori/missing, visualisasi top-N, dan pruning. |
| Natanael Imandatua Manurung | 13524021 | Data, cleaning, preprocessing, feature engineering, KNN, dan submission Kaggle. |
| Rainaldi Pratama Sembiring | 13524117 | SVM/SMO, kernel Linear/Polynomial/Gaussian, kalibrasi, serta visualisasi margin dan support vector. |
| Raynard Fausta | 13524052 | Logistic Regression, L1/L2/Elastic Net, evaluation.py, optimizer, dan animasi log-loss. |
