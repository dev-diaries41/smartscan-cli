import os
import hashlib
from typing import List
from dataclasses import dataclass

from smartscan.organiser.analyser import FileAnalyser
from smartscan.processor.processor import BatchProcessor
from smartscan.processor.processor_listener import ProcessorListener


@dataclass
class ClassificationResult:
    item: str
    class_id: str
    similarity: float

class FileScanner(BatchProcessor[str, ClassificationResult]):
    def __init__(self, 
                 analyser: FileAnalyser,
                 destination_dirs: List[str],
                 listener = ProcessorListener[str, ClassificationResult],
                 **kwargs
                 ):
        super().__init__(listener=listener, **kwargs)
        self.analyser = analyser
        self.destination_dirs = destination_dirs

    def on_process(self, item):
        if not os.path.isfile(item):
            raise ValueError(f"Invalid file: {item}")
        
        destination_dir, best_similarity = self.analyser.compare_file_to_dirs(item, self.destination_dirs)
        
        if best_similarity <= self.analyser.similarity_threshold:
            raise ValueError("Below threshold")

        return ClassificationResult(item, destination_dir, best_similarity)
             
    
    async def on_batch_complete(self, batch):
        self.listener.on_batch_complete(batch)
    
    @staticmethod
    def _hash_string(s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()
