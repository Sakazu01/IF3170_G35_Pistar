"""Metrik evaluasi untuk klasifikasi biner (0 = tidak batal, 1 = batal).

Evaluasi bukan bagian yang wajib dibuat from scratch, jadi semua metrik memakai scikit-learn.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

# Analisis cost-sensitive: false positive (booking dikira batal) dua kali lebih mahal dari false negative.
COST_FP = 2.0
COST_FN = 1.0


def confusion_counts(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def pr_auc(y, proba):
    # Luas di bawah kurva precision-recall dengan aturan trapesium, bukan average precision.
    precision, recall, _ = precision_recall_curve(y, proba)
    return float(auc(recall, precision))


def evaluate(y, proba, threshold=0.5):
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_counts(y, pred)
    precision = float(precision_score(y, pred, zero_division=0))
    recall = float(recall_score(y, pred, zero_division=0))
    f1 = float(f1_score(y, pred, zero_division=0))
    has_both_classes = 0 < y.sum() < len(y)
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'precision_positive': precision, 'recall_positive': recall, 'f1_positive': f1,
        'macro_f1': float(f1_score(y, pred, labels=[0, 1], average='macro', zero_division=0)),
        'precision': precision, 'recall': recall, 'f1': f1,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'roc_auc': float(roc_auc_score(y, proba)) if has_both_classes else float('nan'),
        'pr_auc': pr_auc(y, proba) if y.any() else float('nan'),
        'expected_cost': (COST_FP * fp + COST_FN * fn) / len(y),
    }


def threshold_curve(y, proba, thresholds=None):
    """Confusion matrix dan metrik untuk banyak threshold sekaligus.

    Tanpa `thresholds`, semua titik potong yang mungkin dicoba: setiap peluang unik, 0, 0.5,
    dan satu nilai tepat di atas peluang terbesar (semua booking diprediksi tidak batal).
    """
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    if thresholds is None:
        thresholds = np.unique(np.r_[0.0, proba, 0.5, np.nextafter(proba.max(), np.inf)])
    curve = pd.DataFrame([(t, *confusion_counts(y, proba >= t)) for t in thresholds],
                         columns=['threshold', 'tn', 'fp', 'fn', 'tp'])
    tn, fp, fn, tp = curve.tn, curve.fp, curve.fn, curve.tp
    curve['accuracy'] = (tp + tn) / len(y)
    curve['precision'] = tp / np.maximum(tp + fp, 1)
    curve['recall'] = tp / np.maximum(tp + fn, 1)
    curve['f1'] = curve['f1_positive'] = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    curve['macro_f1'] = tp / np.maximum(2 * tp + fp + fn, 1) + tn / np.maximum(2 * tn + fp + fn, 1)
    curve['cost'] = COST_FP * fp + COST_FN * fn
    curve['expected_cost'] = curve.cost / len(y)
    return curve
