"""
unlabeled_logreg.py
Logistic regression with missing labels.

Class: UnlabeledLogReg

Two algorithms:
  - 'label_propagation' : train model on labeled data, predict unlabeled, combine
  - 'em'                : iterative completion with EM (Expectation-Maximization)
"""

import numpy as np
from fista import FISTASelector


class UnlabeledLogReg:
    """
    Parameters
    method : str
        'label_propagation' or 'em'
    lambdas : array-like or None
        Lambda values to try for FISTA.
    measure : str
        Metric used for lambda selection.
    max_iter_em : int
        Maximum number of iterations for EM algorithm.
    tol_em : float
        Convergence tolerance for EM.
    max_iter_fista : int
        Maximum number of iterations for FISTA.
    random_state : int or None
    """

    def __init__(
            self,
            method='label_propagation',
            lambdas=None,
            measure='f1',
            max_iter_em=20,
            tol_em=1e-3,
            max_iter_fista=1000,
            random_state=42,
    ):
        if method not in ('label_propagation', 'em'):
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose 'label_propagation' or 'em'."
            )
        self.method = method
        self.lambdas = lambdas if lambdas is not None else np.logspace(-4, 1, 20)
        self.measure = measure
        self.max_iter_em = max_iter_em
        self.tol_em = tol_em
        self.max_iter_fista = max_iter_fista
        self.random_state = random_state
        self.selector = None   # trained FISTASelector

    # Main fit function

    def fit(self, X, y_obs, X_valid, y_valid):

        X = np.asarray(X, dtype=float)
        y_obs = np.asarray(y_obs, dtype=float)

        if self.method == 'label_propagation':
            y_complete = self._label_propagation(X, y_obs, X_valid, y_valid)
        else:
            y_complete = self._em(X, y_obs, X_valid, y_valid)

        # Final FISTA training with completed Y
        self.selector = FISTASelector(
            lambdas=self.lambdas,
            max_iter=self.max_iter_fista,
        )
        self.selector.fit(X, y_complete, X_valid, y_valid, measure=self.measure)
        return self

    # Algorithm 1: Label Propagation

    def _label_propagation(self, X, y_obs, X_valid, y_valid):
        """
        Step 1: Train a model using only labeled observations.
        Step 2: Predict missing labels with this model.
        Step 3: Combine predictions with true labels.

        """
        labeled_mask = (y_obs != -1)
        unlabeled_mask = (y_obs == -1)

        X_labeled = X[labeled_mask]
        y_labeled = y_obs[labeled_mask]

        # Step 1: Train model on labeled data
        init_selector = FISTASelector(
            lambdas=self.lambdas,
            max_iter=self.max_iter_fista,
        )
        init_selector.fit(X_labeled, y_labeled, X_valid, y_valid, measure=self.measure)

        # Step 2: Generate predictions for unlabeled observations
        y_complete = y_obs.copy()
        if unlabeled_mask.sum() > 0:
            proba = init_selector.predict_proba(X[unlabeled_mask])
            # If probability > 0.5 predict 1, else 0
            y_complete[unlabeled_mask] = (proba >= 0.5).astype(float)

        return y_complete

    # Algorithm 2: EM (Expectation-Maximization)

    def _em(self, X, y_obs, X_valid, y_valid):

        labeled_mask = (y_obs != -1)
        unlabeled_mask = (y_obs == -1)

        # Initialization: get first estimate using label propagation
        y_current = self._label_propagation(X, y_obs, X_valid, y_valid)

        rng = np.random.default_rng(self.random_state)

        for iteration in range(self.max_iter_em):

            # M step: train model with current labels
            selector = FISTASelector(
                lambdas=self.lambdas,
                max_iter=self.max_iter_fista,
            )
            selector.fit(X, y_current, X_valid, y_valid, measure=self.measure)

            # E step: compute new probabilities for unlabeled observations
            if unlabeled_mask.sum() > 0:
                proba = selector.predict_proba(X[unlabeled_mask])
                y_new = y_current.copy()
                y_new[unlabeled_mask] = (proba >= 0.5).astype(float)
            else:
                break

            # Convergence check: did labels change?
            n_changed = (y_new[unlabeled_mask] != y_current[unlabeled_mask]).sum()
            y_current = y_new

            if n_changed == 0:
                print(f"  EM converged at iteration {iteration+1}.")
                break

        return y_current

    # Prediction methods

    def predict_proba(self, X):
        """Probability of positive class."""
        if self.selector is None:
            raise RuntimeError("fit() must be called first.")
        return self.selector.predict_proba(X)

    def predict(self, X, threshold=0.5):
        """Binary prediction."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def validate(self, X, y, measure):
        """Validation/test metric."""
        if self.selector is None:
            raise RuntimeError("fit() must be called first.")
        return self.selector.best_model.validate(X, y, measure)
