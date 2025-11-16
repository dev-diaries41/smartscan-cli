import numpy as np
from tqdm import tqdm
from smartscan.processor.processor_listener import ProcessorListener

# TODO: Websockets / SSE with desktop app
class FileIndexerListener(ProcessorListener[str, tuple[str, np.ndarray]]):
    def on_active(self):
        self.pbar = tqdm(total=100, desc="Indexing")
        
    def on_progress(self, progress):
        self.pbar.n = int(progress * 100)
        self.pbar.refresh()
        
    def on_fail(self, result):
        self.pbar.close()
        print(result.error)

    def on_complete(self, result):
        self.pbar.close()
        print(f"Results -  Total processed: {result.total_processed} | Time elapsed: {result.time_elapsed:.4f}s")

