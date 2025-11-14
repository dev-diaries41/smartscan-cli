from abc import ABC, abstractmethod
from typing import Generic
from smartscan.processor.types import Input, Output, MetricsFailure, MetricsSuccess


class ProcessorListener(ABC, Generic[Input, Output]):
    def on_active(self):
        pass
    def on_progress(self, progress: float):
        pass
    def on_complete(self, result: MetricsSuccess):
        pass
    def on_batch_complete(self, batch: list[Output]):
        pass
    def on_error(self, e: Exception, item: Input):
        pass
    def on_fail(self, result: MetricsFailure):
        pass