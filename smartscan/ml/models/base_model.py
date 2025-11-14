from abc import abstractmethod, ABC

class BaseModel(ABC):
    @abstractmethod
    def run(self):
        pass
    @abstractmethod
    def load(self):
        pass
    @abstractmethod
    def is_load(self) -> bool:
        pass
    @abstractmethod
    def close(self):
        pass
