from typing import TypeVar
from dataclasses import dataclass

Input = TypeVar("Input")
Output = TypeVar("Output")

@dataclass
class MetricsSuccess:
    total_processed: int = 0
    time_elapsed: float = 0.0

@dataclass
class MetricsFailure:
    total_processed: int
    time_elapsed: int
    error: Exception
    