import os
import uuid
from typing import List
from smartscan.organiser.analyser import FileAnalyser
from smartscan.processor.processor import BatchProcessor
from smartscan.utils.file import move_file
from smartscan.data.scan_history import ScanHistoryDB, ScanHistory


class FileScanner(BatchProcessor[str, tuple[str, str]]):
    def __init__(self, 
                 analyser: FileAnalyser,
                 destination_dirs: List[str],
                 db_path: str
                 ):
        self.analyser = analyser
        self.destination_dirs = destination_dirs
        self.scan_history_db = ScanHistoryDB(db_path)

    def on_process(self, item):
        soruce_file = item

        if os.path.isfile(soruce_file):
            raise ValueError(f"Invalid file: {soruce_file}")
        
        destination_dir, best_similarity = self.analyser.compare_file_to_dirs(soruce_file, self.destination_dirs)
                    
        if best_similarity >= self.similarity_threshold:
            raise ValueError("Below threshold")

        new_path = move_file(soruce_file, destination_dir)

        if new_path is None:
            raise ValueError("Failed to move file")
        
        return soruce_file, str(new_path)
             
    
    async def on_batch_complete(self, batch):
        scans = [ScanHistory(scan_id=str(uuid.uuid4()), source_file=move_info[0], destination_file=move_info[1]) for move_info in batch]
        self.scan_history_db.add(scans)
