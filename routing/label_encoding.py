"""Sklearn helper: string agent labels → integer for MLP / early stopping."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder


class LabelEncodingClassifier(BaseEstimator, ClassifierMixin):
    """Encode string labels for estimators that need integer ``y`` (e.g. MLP early stopping)."""

    _estimator_type = "classifier"

    def __init__(self, estimator: Any | None = None):
        self.estimator = estimator

    def __sklearn_tags__(self):
        from sklearn.utils._tags import ClassifierTags

        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags(multi_class=True)
        return tags

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: np.ndarray | None = None,
    ) -> LabelEncodingClassifier:
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(np.asarray(y, dtype=str))
        est = clone(self.estimator)
        if sample_weight is not None:
            est.fit(X, y_enc, sample_weight=sample_weight)
        else:
            est.fit(X, y_enc)
        self.estimator_ = est
        self.classes_ = self.le_.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        pred = self.estimator_.predict(X)
        return self.le_.inverse_transform(pred)

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator_.predict_proba(X)
