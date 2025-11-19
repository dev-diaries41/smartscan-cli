from abc import abstractmethod, ABC
from typing import TypeVar, Generic
from PIL import Image
import numpy as np

T = TypeVar("T")

class EmbeddingProvider(ABC, Generic[T]):
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass
    @abstractmethod
    def embed(self, data: T) -> np.ndarray:
        pass
    @abstractmethod
    def embed_batch(self, data: list[T]) -> np.ndarray:
        pass

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def is_initialized(self) -> bool: 
        pass
    
    @abstractmethod
    def close_session(self):
        pass

ImageEmbeddingProvider = EmbeddingProvider[Image.Image]
TextEmbeddingProvider = EmbeddingProvider[str]