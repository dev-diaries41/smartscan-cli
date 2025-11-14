import os
from typing import List
from organiser.analyser import FileAnalyser
from processor.processor import BatchProcessor
from utils.file import move_file

class FileScanner(BatchProcessor[str, tuple[str, str]]):
    def __init__(self, 
                 analyser: FileAnalyser,
                 destination_dirs: List[str]
                 ):
        self.analyser = analyser
        self.destination_dirs = destination_dirs

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
             
    
    def on_batch_complete(self, batch):
        # TODO add move history to sqlite db
        return super().on_batch_complete(batch)
 