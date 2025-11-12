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
    def embedBatch(self, data: list[Any]) -> np.ndarray:
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def isInitialized(self) -> bool: 
        pass
    
    @abstractmethod
    def closeSession(self):
        pass