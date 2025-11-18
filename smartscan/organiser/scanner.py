import os
import uuid
import hashlib
import shutil
from typing import List
from dataclasses import dataclass

from smartscan.organiser.analyser import FileAnalyser
from smartscan.processor.processor import BatchProcessor
from smartscan.data.scan_history import ScanHistoryDB, ScanHistory


@dataclass
class ClassificationResult:
    item: str
    class_id: str
    similarity: float

class FileScanner(BatchProcessor[str, ClassificationResult]):
    def __init__(self, 
                 analyser: FileAnalyser,
                 destination_dirs: List[str],
                 db_path: str,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.analyser = analyser
        self.destination_dirs = destination_dirs
        self.scan_history_db = ScanHistoryDB(db_path)

    def on_process(self, item):
        if not os.path.isfile(item):
            raise ValueError(f"Invalid file: {item}")
        
        destination_dir, best_similarity = self.analyser.compare_file_to_dirs(item, self.destination_dirs)
        
        if best_similarity <= self.analyser.similarity_threshold:
            raise ValueError("Below threshold")

        return ClassificationResult(item, destination_dir, best_similarity)
             
    
    async def on_batch_complete(self, batch):
        # TODO store assigned class ids
        pass
        # scans = [ScanHistory(scan_id=str(uuid.uuid4()), file_id=self._hash_string(move_info[0]), source_file=move_info[0], destination_file=move_info[1]) for move_info in batch]
        # self.scan_history_db.add(scans)
        print(batch)
    
    @staticmethod
    def _hash_string(s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()
