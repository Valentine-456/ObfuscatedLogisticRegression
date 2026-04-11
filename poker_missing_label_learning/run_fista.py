"""
run_fista.py
============
FISTA implementasyonunu poker_processed.csv üzerinde test eder.
Lambda seçimi validation setinde yapılır.
Sonuçlar sklearn ile karşılaştırılır.

Kullanım:
    python run_fista.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import roc_auc_score, f1_score
from fista import LogisticRegressionFISTA, FISTASelector

# -------------------------------------------------------------------
# 1. Veriyi yükle
# -------------------------------------------------------------------

df = pd.read_csv("poker_data_preprocessed.csv")

X = df.drop(columns=["result"]).to_numpy()
y = df["result"].to_numpy()

print(f"Dataset: {X.shape[0]} satır, {X.shape[1]} feature")
print(f"Pozitif sınıf oranı: {y.mean():.2f}\n")

# -------------------------------------------------------------------
# 2. Train / Validation / Test split
# -------------------------------------------------------------------
# Önce %80 train+valid, %20 test
# Sonra train+valid'i %75/%25 olarak böl → %60 train, %20 valid, %20 test

X_trainval, X_test, y_trainval, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_valid, y_train, y_valid = train_test_split(
	X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
)

print(f"Train:      {X_train.shape[0]} satır")
print(f"Validation: {X_valid.shape[0]} satır")
print(f"Test:       {X_test.shape[0]} satır\n")

# -------------------------------------------------------------------
# 3. Lambda seçimi — validation setinde F1 optimize et
# -------------------------------------------------------------------

MEASURE = 'f1'
LAMBDAS = np.logspace(-4, 1, 30)

print(f"Lambda aralığı: {LAMBDAS[0]:.5f} — {LAMBDAS[-1]:.1f}")
print(f"Metrik: {MEASURE}\n")

selector = FISTASelector(lambdas=LAMBDAS, max_iter=1000, tol=1e-5)
selector.fit(X_train, y_train, X_valid, y_valid, measure=MEASURE)

print(f"En iyi lambda: {selector.best_lambda:.6f}")
print(f"Validation {MEASURE}: {selector.scores[selector.best_lambda]:.4f}\n")

# -------------------------------------------------------------------
# 4. Test seti değerlendirmesi — FISTA
# -------------------------------------------------------------------

print("=== FISTA — Test Seti ===")
best_model = selector.best_model

for metric in ['recall', 'precision', 'f1', 'balanced_accuracy', 'roc_auc', 'pr_auc']:
	score = best_model.validate(X_test, y_test, metric)
	print(f"  {metric:20s}: {score:.4f}")

w = best_model.w[1:]   # bias hariç
print(f"\nSıfır katsayı sayısı: {(w == 0).sum()} / {len(w)}")
print(f"Sıfır olmayan katsayılar: {(w != 0).sum()} / {len(w)}")

# -------------------------------------------------------------------
# 5. sklearn karşılaştırma
# -------------------------------------------------------------------

print("\n=== sklearn (L1) — Test Seti ===")

# sklearn'de C = 1 / (lambda * n_train)
C_best = 1.0 / (selector.best_lambda * len(y_train))

sklearn_model = SklearnLR(
	C=C_best,
	solver='saga',
	l1_ratio=1.0,       # penalty='l1' yerine
	max_iter=2000,
	random_state=42
)
sklearn_model.fit(X_train, y_train)
sk_proba = sklearn_model.predict_proba(X_test)[:, 1]
sk_pred  = (sk_proba >= 0.5).astype(int)

print(f"  {'f1':20s}: {f1_score(y_test, sk_pred):.4f}")
print(f"  {'roc_auc':20s}: {roc_auc_score(y_test, sk_proba):.4f}")
sk_w = sklearn_model.coef_[0]
print(f"\nSıfır katsayı sayısı: {(sk_w == 0).sum()} / {len(sk_w)}")

# -------------------------------------------------------------------
# 6. Katsayı karşılaştırma tablosu
# -------------------------------------------------------------------

feature_names = df.drop(columns=["result"]).columns.tolist()
print("\n=== Katsayı Karşılaştırması ===")
print(f"{'Feature':15s}  {'FISTA':>10s}  {'sklearn':>10s}")
print("-" * 40)
for name, w_fista, w_sk in zip(feature_names, best_model.w[1:], sklearn_model.coef_[0]):
	print(f"{name:15s}  {w_fista:10.4f}  {w_sk:10.4f}")

# -------------------------------------------------------------------
# 7. Grafikler
# -------------------------------------------------------------------

selector.plot(measure=MEASURE)
selector.plot_coefficients()

print("\nTamamlandı.")

