from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional

class BaseRerankStrategy(ABC):
    """Abstract interface for rerank strategies."""
    @abstractmethod
    def apply(self, df: pd.DataFrame, *, context: Optional[dict] = None) -> pd.DataFrame:
        ...


class NoOpStrategy(BaseRerankStrategy):
    """Pass-through (no reranking)."""
    def apply(self, df: pd.DataFrame, *, context: Optional[dict] = None) -> pd.DataFrame:
        return df