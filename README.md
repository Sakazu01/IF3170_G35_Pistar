# IF3170_G35_Pistar

Tugas Besar 2 IF3170 Inteligensi Artifisial: memprediksi apakah sebuah booking penginapan dibatalkan (`is_canceled`). Lima algoritma diimplementasikan from scratch dengan NumPy (KNN, Decision Tree, Logistic Regression, SVM, dan Naive Bayes), lalu dibandingkan dengan scikit-learn. Metrik kompetisi Kaggle adalah Macro F1.

Model final yang dipakai untuk submission adalah KNN from scratch (jarak Manhattan, k = 11) dengan target encoding yang dihitung di dalam data training dan bobot kolom dari mutual information.

## Struktur repository

```
src/
  base.py            kelas induk model dan pemeriksaan masukan
  evaluation.py      metrik evaluasi
  models/            KNN, DecisionTree, LogisticRegression, SVM, NaiveBayes from scratch
  bonus/             pruning dan visualisasi pohon, perbandingan optimizer, video log-loss,
                     plot kernel SVM, graf Naive Bayes, analisis cost-sensitive
notebooks/main.ipynb alur lengkap: pemahaman data, preprocessing, eksperimen, tuning,
                     perbandingan dengan scikit-learn, dan submission Kaggle
output/              hasil run: submission.csv, gambar, tabel, dan animasi
docs/                laporan (PDF)
data/                tempat dataset Kaggle (tidak di-commit)
```

## Setup

Butuh Python 3.12. Versi library dikunci di `requirements.txt` agar hasil run sama persis dengan submisi Kaggle.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; di Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Unduh `train.csv`, `test.csv`, dan `sample_submission.csv` dari halaman kompetisi Kaggle, lalu taruh di folder `data/`.

## Menjalankan program

Buka `notebooks/main.ipynb` di Jupyter, lalu jalankan semua sel (Run All). Bisa juga lewat terminal dari folder utama:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/main.ipynb
```

Satu run penuh memakan waktu sekitar 30 menit karena tuning mencoba banyak konfigurasi dengan cross-validation. Hasil utamanya adalah `output/submission.csv`. Seed dikunci di 42 dan perhitungan dibatasi satu thread, sehingga setiap run menghasilkan submission yang sama.

## Model Pilihan Dan Hasil

Model final adalah **KNN from scratch** dengan `k=11`, jarak **Manhattan**, bobot **invers jarak**, dan threshold `0.49`. Model menggunakan fitur dasar dan tanggal, target encoding, serta bobot mutual information (MI). KNN dipilih karena memperoleh mean **Macro F1 tertinggi pada konfirmasi 5-fold CV development, yaitu 0.8383**, di antara kandidat yang diuji. Threshold dipilih dari prediksi out-of-fold (OOF) development.

Skor Kaggle yang diperoleh adalah **0.84092**.

## Pembagian tugas

| NIM | Nama | Tanggung jawab |
|-----|------|----------------|
| 13524021 | Natanael Imandatua Manurung | Pengolahan data, cleaning, preprocessing, feature engineering, KNN, dan pembuatan submission Kaggle. |
| 13524107 | Rava Khoman Tuah Saragih | Decision Tree C4.5, penanganan kategori/missing, visualisasi pohon top-N, dan pruning. |
| 13524052 | Raynard Fausta | Logistic Regression, regularisasi L1/L2/Elastic Net, evaluation.py, perbandingan optimizer, dan animasi log-loss. |
| 13524117 | Rainaldi Pratama Sembiring | SVM dengan SMO, kernel Linear/Polynomial/Gaussian, kalibrasi, serta visualisasi margin dan support vector. |
| 13524085 | Ariel Cornelius Sitorus | Naive Bayes, graf hubungan kelas-fitur, serta integrasi model, bonus, dan notebook. |
