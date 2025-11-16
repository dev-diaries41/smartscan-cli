import os
import uuid
import hashlib
from typing import List
from smartscan.organiser.analyser import FileAnalyser
from smartscan.processor.processor import BatchProcessor
from smartscan.utils.file import move_file
from smartscan.data.scan_history import ScanHistoryDB, ScanHistory


class FileScanner(BatchProcessor[str, tuple[str, str]]):
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
        source_file = item

        if not os.path.isfile(source_file):
            raise ValueError(f"Invalid file: {source_file}")
        
        dir_similarities_dict = self.analyser.compare_file_to_dirs(source_file, self.destination_dirs)
        dir_similarities = sorted(dir_similarities_dict.items())
        destination_dir, best_similarity = dir_similarities[0]
                    
        if best_similarity >= self.analyser.similarity_threshold:
            raise ValueError("Below threshold")

        new_path = move_file(source_file, destination_dir)

        if new_path is None:
            raise ValueError("Failed to move file")
        
        return source_file, str(new_path)
             
    
    async def on_batch_complete(self, batch):
        scans = [ScanHistory(scan_id=str(uuid.uuid4()), file_id=self._hash_string(move_info[0]), source_file=move_info[0], destination_file=move_info[1]) for move_info in batch]
        self.scan_history_db.add(scans)
    
    @staticmethod
    def _hash_string(s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()
