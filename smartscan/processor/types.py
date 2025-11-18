from dataclasses import dataclass
from typing import TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")

@dataclass
class MetricsSuccess:
    total_processed: int = 0
    time_elapsed: float = 0.0

@dataclass
class MetricsFailure:
    total_processed: int
    time_elapsed: float
    error: Exception
    
