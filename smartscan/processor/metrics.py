from dataclasses import dataclass

@dataclass
class MetricsSuccess:
    total_processed: int = 0
    time_elapsed: float = 0.0

@dataclass
class MetricsFailure:
    total_processed: int
    time_elapsed: float
    error: Exception
    
