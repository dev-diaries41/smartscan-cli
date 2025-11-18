from tqdm import tqdm

from smartscan.processor.processor_listener import ProcessorListener
from smartscan.organiser.classifier import ClassificationResult

#  Placeholder for dev
class FileClassifierListener(ProcessorListener[str, ClassificationResult]):
    def __init__(self):
        self.progress_bar = tqdm(total=100, desc="Scanning")

    async def on_progress(self, progress):
        self.progress_bar.n = int(progress * 100)
        self.progress_bar.refresh()
        
    async def on_fail(self, result):
        self.progress_bar.close()
        print(result.error)

    async def on_complete(self, result):
        self.progress_bar.close()
        print(f"Results -  Total processed: {result.total_processed} | Time elapsed: {result.time_elapsed:.4f}s")

