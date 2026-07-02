from abc import ABC, abstractmethod
from typing import List, Optional
from smartscan.embeds.types import QueryResult, StoredEmbedding
from numpy.typing import NDArray


class EmbeddingStore(ABC):
    @abstractmethod
    def add(self, items: List[StoredEmbedding]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, ids: Optional[List[str]] = None) -> List[StoredEmbedding]:
        raise NotImplementedError

    @abstractmethod
    def query(self, query_embed: NDArray, topK: int, ids: Optional[List[str]] = None, include_sims: bool = False ) -> QueryResult:
        raise NotImplementedError

    @abstractmethod
    def update(self, items: List[StoredEmbedding]) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, items: List[StoredEmbedding]) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError
