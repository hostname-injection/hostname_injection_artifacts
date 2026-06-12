from __future__ import annotations

from typing import Iterable, List, Optional, Sequence


class BaselineModel:
    name: str = "baseline"

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, texts: Sequence[str], batch_size: Optional[int] = None) -> List[int]:  # pragma: no cover - interface
        raise NotImplementedError

    def predict_scores(self, texts: Sequence[str], batch_size: Optional[int] = None) -> Optional[Iterable[float]]:
        return None
