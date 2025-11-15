import numpy as np

from smartscan.processor.processor_listener import ProcessorListener

# TODO: Websockets / SSE with desktop app
class FileIndexerListener(ProcessorListener[str, tuple[str, np.ndarray]]):
    def on_active(self):
        print("Indexing starting...")
    def on_progress(self, progress):
        print(f"Progress: {100 * progress:.2f}%")
    def on_fail(self, result):
        print(result.error)