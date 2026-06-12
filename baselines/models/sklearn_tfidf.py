from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np

from .base import BaselineModel


class TfidfClassifier(BaselineModel):
    def __init__(
        self,
        *,
        model_type: str = "logreg",
        ngram_range: tuple[int, int] = (3, 3),
        max_features: int = 200_000,
        min_df: int = 1,
        class_weight: str = "balanced",
        random_state: int = 13,
    ) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import Pipeline
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "scikit-learn is required for TF-IDF baselines. Install with conda: "
                "conda install -c conda-forge scikit-learn"
            ) from exc

        self.model_type = model_type
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.class_weight = class_weight
        self.random_state = random_state
        self._pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=ngram_range,
                        max_features=max_features,
                        min_df=min_df,
                        lowercase=True,
                    ),
                ),
                ("clf", self._build_model(model_type)),
            ]
        )

    def _build_model(self, model_type: str):
        model_type = model_type.lower()
        if model_type == "logreg":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                max_iter=2000,
                class_weight=self.class_weight,
                n_jobs=-1,
                solver="lbfgs",
                random_state=self.random_state,
            )
        if model_type == "linear_svm":
            from sklearn.svm import LinearSVC

            return LinearSVC(class_weight=self.class_weight, random_state=self.random_state)
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                n_jobs=-1,
                random_state=self.random_state,
            )
        if model_type == "extra_trees":
            from sklearn.ensemble import ExtraTreesClassifier

            return ExtraTreesClassifier(
                n_estimators=400,
                max_features="sqrt",
                n_jobs=-1,
                random_state=self.random_state,
            )
        if model_type == "isolation_forest":
            from sklearn.ensemble import IsolationForest

            return IsolationForest(
                n_estimators=300,
                contamination="auto",
                random_state=self.random_state,
                n_jobs=-1,
            )
        if model_type == "ocsvm":
            from sklearn.svm import OneClassSVM

            return OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        if model_type == "xgboost":
            try:
                import xgboost as xgb
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "xgboost is required for the XGBoost baseline. Install with conda: "
                    "conda install -c conda-forge xgboost"
                ) from exc

            return xgb.XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                tree_method="hist",
                random_state=self.random_state,
            )
        raise ValueError(f"Unsupported model type: {model_type}")

    @property
    def _is_unsupervised(self) -> bool:
        return self.model_type in {"isolation_forest", "ocsvm"}

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        if self._is_unsupervised:
            benign = [t for t, y in zip(texts, labels) if int(y) == 0]
            if not benign:
                benign = list(texts)
            self._pipeline.fit(benign)
        else:
            self._pipeline.fit(texts, labels)

    def predict(self, texts: Sequence[str], batch_size: Optional[int] = None) -> List[int]:
        preds = self._pipeline.predict(texts)
        preds = np.asarray(preds)
        if self._is_unsupervised:
            # IsolationForest/OneClassSVM return 1 for inliers, -1 for outliers
            return (preds == -1).astype(int).tolist()
        return preds.astype(int).tolist()

    def predict_scores(self, texts: Sequence[str], batch_size: Optional[int] = None) -> Optional[Iterable[float]]:
        clf = self._pipeline.named_steps.get("clf")
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(self._pipeline.named_steps["tfidf"].transform(texts))
            return np.asarray(scores).tolist()
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(self._pipeline.named_steps["tfidf"].transform(texts))
            return np.asarray(probs)[:, 1].tolist()
        return None
