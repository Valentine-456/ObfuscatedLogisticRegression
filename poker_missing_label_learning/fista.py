"""
fista.py
========
L1 Regülarizasyonlu Lojistik Regresyon — FISTA implementasyonu.

Sınıf: LogisticRegressionFISTA
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score
)


class LogisticRegressionFISTA:
    """
    FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) ile
    L1 regülarizasyonlu lojistik regresyon.

    Parameters
    ----------
    lambda_val : float
        L1 ceza katsayısı. Büyük değer → daha seyrek model.
    max_iter : int
        Maksimum iterasyon sayısı.
    tol : float
        Yakınsama toleransı. İki iterasyon arası değişim bunun altına
        düşünce durur.
    """
    
    def __init__(self, lambda_val=1.0, max_iter=1000, tol=1e-4):
        self.lambda_val = lambda_val
        self.max_iter = max_iter
        self.tol = tol
        self.w = None       # katsayılar (bias dahil)
        self.L = None       # Lipschitz sabiti — adım boyutunu belirler
    
    # ------------------------------------------------------------------
    # Yardımcı fonksiyonlar
    # ------------------------------------------------------------------
    
    def _sigmoid(self, z):
        """
        Numerically stable sigmoid.

        Sıradan 1/(1+exp(-z)) büyük negatif z'de overflow verir.
        np.where ile z'nin işaretine göre iki farklı formül kullanırız —
        matematiksel sonuç aynı, ama overflow olmaz.
        """
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z))
        )
    
    def _soft_thresholding(self, w, threshold):
        """
        L1 normunun proximal operatörü — soft thresholding.

        |w_j| <= threshold ise → 0 yap (Lasso etkisi)
        |w_j| >  threshold ise → threshold kadar sıfıra çek

        DİKKAT: Bias terimi (w[0]) regülarize edilmez!
        Bias'a shrinkage uygulamak modeli bozar.
        """
        w_thresh = np.sign(w) * np.maximum(np.abs(w) - threshold, 0.0)
        w_thresh[0] = w[0]   # bias'ı geri koy — dokunmadık
        return w_thresh
    
    def _compute_lipschitz(self, X_design):
        """
        Loss fonksiyonunun Lipschitz sabiti.

        Logistic regression için: L = ||X||_F^2 / (4 * n)

        Bu sabit adım boyutunu belirler: step_size = 1/L
        Büyük L → küçük adım → yavaş ama garantili yakınsama.
        """
        n = X_design.shape[0]
        return np.linalg.norm(X_design, 'fro') ** 2 / (4 * n)
    
    def _gradient(self, X_design, y, w):
        """
        Logistic loss'un gradyanı.

        grad = (1/n) * X^T * (sigmoid(X*w) - y)

        Türetme:
          Loss = -mean[ y*log(p) + (1-y)*log(1-p) ]
          dLoss/dw = (1/n) * X^T * (p - y)
          burada p = sigmoid(X*w)
        """
        n = X_design.shape[0]
        p = self._sigmoid(X_design @ w)
        return (X_design.T @ (p - y)) / n
    
    # ------------------------------------------------------------------
    # Ana FISTA döngüsü
    # ------------------------------------------------------------------
    
    def fit(self, X, y):
        """
        FISTA ile modeli eğit.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Eğitim feature matrisi.
        y : array (n_samples,)
            Binary etiketler (0 veya 1).
        """
        n_samples, n_features = X.shape
        
        # Bias sütunu ekle: X_design = [1, x1, x2, ...]
        X_design = np.c_[np.ones(n_samples), X]
        
        # Lipschitz sabiti — sabit adım boyutu
        self.L = self._compute_lipschitz(X_design)
        step = 1.0 / self.L         # adım boyutu
        
        # Başlangıç noktaları
        w_k = np.zeros(n_features + 1)   # mevcut katsayılar
        z_k = w_k.copy()                  # momentum arama noktası
        t_k = 1.0                         # momentum katsayısı
        
        for i in range(self.max_iter):
            
            # 1. z_k noktasında gradyan hesapla
            grad = self._gradient(X_design, y, z_k)
            
            # 2. Gradyan adımı
            v = z_k - step * grad
            
            # 3. Soft thresholding — L1 cezasını uygula
            #    threshold = lambda * step
            w_next = self._soft_thresholding(v, self.lambda_val * step)
            
            # 4. Yakınsama kontrolü
            #    İki iterasyon arası değişim tol'dan küçükse dur
            if np.linalg.norm(w_next - w_k) < self.tol:
                break
            
            # 5. Momentum güncelleme (Nesterov)
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
            z_k = w_next + (t_k - 1.0) / t_next * (w_next - w_k)
            
            # 6. Bir sonraki iterasyona geç
            w_k = w_next
            t_k = t_next
        
        self.w = w_k
        return self
    
    # ------------------------------------------------------------------
    # Tahmin
    # ------------------------------------------------------------------
    
    def predict_proba(self, X):
        """
        Her gözlem için pozitif sınıf olasılığı döndür.

        Parameters
        ----------
        X : array (n_samples, n_features)

        Returns
        -------
        proba : array (n_samples,)  — değerler [0, 1] arasında
        """
        if self.w is None:
            raise RuntimeError("Önce fit() çağırılmalı.")
        n = X.shape[0]
        X_design = np.c_[np.ones(n), X]
        return self._sigmoid(X_design @ self.w)
    
    def predict(self, X, threshold=0.5):
        """0.5 eşiğiyle binary tahmin."""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    # ------------------------------------------------------------------
    # Validasyon
    # ------------------------------------------------------------------
    
    def validate(self, X_valid, y_valid, measure):
        """
        Modeli validation setinde değerlendir.

        Parameters
        ----------
        X_valid : array
        y_valid : array
        measure : str
            'recall', 'precision', 'f1', 'balanced_accuracy',
            'roc_auc', 'pr_auc'

        Returns
        -------
        score : float
        """
        proba = self.predict_proba(X_valid)
        y_pred = (proba >= 0.5).astype(int)
        
        measures = {
            'recall':            lambda: recall_score(y_valid, y_pred, zero_division=0),
            'precision':         lambda: precision_score(y_valid, y_pred, zero_division=0),
            'f1':                lambda: f1_score(y_valid, y_pred, zero_division=0),
            'balanced_accuracy': lambda: balanced_accuracy_score(y_valid, y_pred),
            'roc_auc':           lambda: roc_auc_score(y_valid, proba),
            'pr_auc':            lambda: average_precision_score(y_valid, proba),
        }
        
        if measure not in measures:
            raise ValueError(
                f"Bilinmeyen metrik '{measure}'. "
                f"Seçenekler: {list(measures.keys())}"
            )
        return measures[measure]()


# ------------------------------------------------------------------
# Lambda seçimi + plot fonksiyonları
# ------------------------------------------------------------------

class FISTASelector:
    """
    Farklı lambda değerleri için FISTA çalıştırır,
    validation setinde en iyi lambda'yı seçer,
    ve sonuçları görselleştirir.

    Parameters
    ----------
    lambdas : array-like
        Denenecek lambda değerleri listesi.
    max_iter : int
    tol : float
    """
    
    def __init__(self, lambdas=None, max_iter=1000, tol=1e-4):
        if lambdas is None:
            self.lambdas = np.logspace(-4, 1, 30)
        else:
            self.lambdas = np.array(lambdas)
        
        self.max_iter = max_iter
        self.tol = tol
        
        self.models = {}        # lambda → model
        self.scores = {}        # lambda → validation score
        self.best_lambda = None
        self.best_model = None
    
    def fit(self, X_train, y_train, X_valid, y_valid, measure='roc_auc'):
        """
        Her lambda için model eğit ve validation skoru hesapla.

        Parameters
        ----------
        X_train, y_train : eğitim verisi
        X_valid, y_valid : validation verisi
        measure : str — hangi metriğe göre lambda seçilsin
        """
        best_score = -np.inf
        
        for lam in self.lambdas:
            model = LogisticRegressionFISTA(
                lambda_val=lam,
                max_iter=self.max_iter,
                tol=self.tol
            )
            model.fit(X_train, y_train)
            score = model.validate(X_valid, y_valid, measure)
            
            self.models[lam] = model
            self.scores[lam] = score
            
            if score > best_score:
                best_score = score
                self.best_lambda = lam
                self.best_model = model
        
        return self
    
    def predict_proba(self, X):
        """En iyi modelle tahmin."""
        if self.best_model is None:
            raise RuntimeError("Önce fit() çağırılmalı.")
        return self.best_model.predict_proba(X)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def plot(self, measure='roc_auc'):
        """
        Lambda değişirken validation metriği nasıl değişiyor?
        """
        if not self.scores:
            raise RuntimeError("Önce fit() çağırılmalı.")
        
        lambdas = self.lambdas
        scores = [self.scores[lam] for lam in lambdas]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogx(lambdas, scores, 'o-', color='#378ADD', linewidth=2, markersize=5)
        ax.axvline(self.best_lambda, color='#E24B4A', linestyle='--',
                   label=f'best λ = {self.best_lambda:.4f}')
        ax.set_xlabel('λ (log scale)')
        ax.set_ylabel(measure)
        ax.set_title(f'Validation {measure} vs λ  —  FISTA')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_coefficients(self):
        """
        Lambda değişirken katsayılar nasıl değişiyor?
        (Regularization path)
        """
        if not self.models:
            raise RuntimeError("Önce fit() çağırılmalı.")
        
        # Her lambda için katsayıları topla (bias hariç)
        coef_matrix = np.array([
            self.models[lam].w[1:]   # w[0] bias, atlıyoruz
            for lam in self.lambdas
        ])
        
        fig, ax = plt.subplots(figsize=(9, 5))
        for j in range(coef_matrix.shape[1]):
            ax.semilogx(self.lambdas, coef_matrix[:, j], linewidth=1.5, alpha=0.8)
        
        ax.axvline(self.best_lambda, color='#E24B4A', linestyle='--',
                   label=f'best λ = {self.best_lambda:.4f}')
        ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
        ax.set_xlabel('λ (log scale)')
        ax.set_ylabel('katsayı değeri')
        ax.set_title('Regularization Path  —  FISTA')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()