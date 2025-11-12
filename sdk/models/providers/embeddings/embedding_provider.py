from abc import abstractmethod, ABC
from typing import Any
import numpy as np

class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def embeddingDim(self) -> int:
        return self.embeddingDim

    @abstractmethod
    def embed(self, data: Any) -> np.ndarray:
        pass
    @abstractmethod
    def embed_batch(self, data: list[Any]) -> np.ndarray:
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