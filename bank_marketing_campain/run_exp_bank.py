"""
Task 3 experiments for Bank Marketing dataset.
All logic lives inside UnlabeledLogReg — this script only loads data and calls it.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from fista_bank import FISTASelector
from unlabeled_logreg_bank import UnlabeledLogReg

PREPROCESSED_FILE = "bank_preprocessed.csv"
RANDOM_STATE = 42

df = pd.read_csv(PREPROCESSED_FILE)
feature_names = [c for c in df.columns if c != "y"]
X = df[feature_names].values.astype(np.float64)
y = df["y"].values.astype(np.float64)

X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=RANDOM_STATE)

print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")
print(f"Positive ratio — train: {y_train.mean():.3f}  test: {y_test.mean():.3f}\n")

# Select best lambda on oracle data
print("Selecting best lambda...")
sel = FISTASelector(lambdas=np.logspace(-4, 1, 20), max_iter=1000, tol=1e-4)
sel.fit(X_train, y_train, X_val, y_val, measure="roc_auc")
BEST_LAMBDA = sel.best_lambda
print(f"Best lambda: {BEST_LAMBDA:.6f}\n")

# Experiment 1: four schemes
print("=" * 60)
print("EXPERIMENT 1: Four missing schemes (c=0.3)")
print("=" * 60)

for method in ["em", "label_propagation"]:
    print(f"\n>>> Method: {method.upper()}")
    model = UnlabeledLogReg(method=method, lambda_val=BEST_LAMBDA, random_state=RANDOM_STATE)
    results = model.run_schemes(X_train, y_train, X_test, y_test, c=0.3, feature_idx=0)
    results.to_csv(f"results_schemes_{method}.csv")
    print(f"Saved to results_schemes_{method}.csv")

# Experiment 2: MCAR sensitivity
print("\n" + "=" * 60)
print("EXPERIMENT 2: MCAR sensitivity vs c")
print("=" * 60)

for method in ["em", "label_propagation"]:
    print(f"\n>>> Method: {method.upper()}")
    model = UnlabeledLogReg(method=method, lambda_val=BEST_LAMBDA, random_state=RANDOM_STATE)
    sens = model.run_mcar_sensitivity(X_train, y_train, X_test, y_test,
                                      c_values=[0.1, 0.2, 0.3, 0.4, 0.5])
    sens.to_csv(f"results_mcar_sensitivity_{method}.csv")
    print(f"Saved to results_mcar_sensitivity_{method}.csv")

print("\nAll experiments done.")
