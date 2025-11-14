import psutil

class MemoryManager():
    def __init__(self, 
                 low_memory_threshold: int = 400,
                 high_memory_threshold: int = 1600,
                 min_concurrency: int = 1,
                 max_concurrency: int = 8,
                 ):
        self.low_memory_threshold = low_memory_threshold
        self.high_memory_threshold = high_memory_threshold
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency

    def get_available_memory(self):
        return self.to_mb(psutil.virtual_memory().available)
    
    def calculate_concurrency(self):
        available_memory = self.get_available_memory()

        if available_memory < self.low_memory_threshold:
            return self.min_concurrency
        if available_memory >= self.high_memory_threshold:
            return self.max_concurrency
        else:
            ratio = ( available_memory - self.low_memory_threshold) / self.high_memory_threshold - self.low_memory_threshold
            return max(self.min_concurrency, int((self.min_concurrency + ratio * (self.max_concurrency - self.min_concurrency))))
    
    @staticmethod
    def to_mb(memory: int):
        return memory / (1024**2)
    
    @staticmethod
    def to_gb(memory: int):
        return memory / (1024**3)