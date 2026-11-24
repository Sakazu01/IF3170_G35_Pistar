# IF3170_G35_Pistar

Tubes 2 IF3170 Inteligensi Artifisial: prediksi pembatalan booking hotel (`is_canceled`) pakai KNN, Decision Tree, Logistic Regression, SVM, dan Naive Bayes 

## Setup

```bash
pip install -r requirements.txt
```

Taruh `train.csv`, `test.csv`, `sample_submission.csv` dari Kaggle di folder `data/` (nggak di-commit).

## Run

Buka `notebooks/main.ipynb`, Run All. Hasilnya ada di `output/submission.csv`.

## Pembagian tugas

| Nama | NIM | Tugas |
|------|-----|-------|
| Ariel Cornelius Sitorus | 13524085 | Naive Bayes, graf kelas-fitur, serta integrasi model, bonus, dan notebook. |
| Rava Khoman Tuah Saragih | 13524107 | Decision Tree C4.5, penanganan kategori/missing, visualisasi top-N, dan pruning. |
| Natanael Imandatua Manurung | 13524021 | Data, cleaning, preprocessing, feature engineering, KNN, dan submission Kaggle. |
| Rainaldi Pratama Sembiring | 13524117 | SVM/SMO, kernel Linear/Polynomial/Gaussian, kalibrasi, serta visualisasi margin dan support vector. |
| Raynard Fausta | 13524052 | Logistic Regression, L1/L2/Elastic Net, evaluation.py, optimizer, dan animasi log-loss. |