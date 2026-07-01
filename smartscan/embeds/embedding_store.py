from abc import ABC, abstractmethod
from typing import Generic, List, Optional
from smartscan.embeds.types import TData, TMetadata , FilterType, ItemEmbedding, GetResult, QueryResult, Include, ItemEmbeddingUpdate
import numpy as np



class EmbeddingStore(ABC, Generic[TData, TMetadata]):
    @abstractmethod
    def add(self, items: List[ItemEmbedding[TData, TMetadata]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[FilterType] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Include = ["metadatas"],
    ) -> GetResult:
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        query_embeds: List[np.ndarray],
        filter: Optional[FilterType] = None,
        limit: int = 10,
        include: Include = ["metadatas"],
    ) -> QueryResult:
        raise NotImplementedError

    @abstractmethod
    def update(self, items: List[ItemEmbeddingUpdate[TData, TMetadata]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, items: List[ItemEmbeddingUpdate[TData, TMetadata]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: Optional[List[str]] = None, filter: Optional[FilterType] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self, filter: Optional[FilterType] = None) -> int:
        raise NotImplementedError
