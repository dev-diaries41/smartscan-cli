import os
import numpy as np
from typing import List
from enum import IntEnum

from smartscan.ml.providers.embeddings.embedding_provider import EmbeddingProvider
from smartscan.utils.file import save_embedding, load_embedding, get_days_since_last_modified, get_files_from_dirs
from smartscan.utils.ml_ops import generate_prototype_embedding


class AnalyserMode(IntEnum):
    TEXT = 0
    IMAGE = 1
    VIDEO = 2

class FileAnalyser:
    def __init__(self, 
                 image_encoder: EmbeddingProvider, 
                 text_encoder: EmbeddingProvider,
                 similarity_threshold: float,
                 max_files_for_prototypes: int = 30,
                 refresh_prototype_duration: int  = 7,
                 ):
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.similarity_threshold = similarity_threshold
        self.valid_img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        self.valid_txt_exts = ('.txt', '.md', '.rst', '.html', '.json')
        self.valid_vid_exts = ('.mp4', '.mkv', '.webm')
        self.max_files_for_prototypes = max_files_for_prototypes
        self.refresh_prototype_duration = refresh_prototype_duration

    def compare_files(self, filepath1: str, filepath2: str):
        """Compute the cosine similarity between two files"""
        is_image_mode = self.are_files_valid(self.valid_img_exts, [filepath1, filepath2])
        is_text_mode = self.are_files_valid(self.valid_txt_exts, [filepath1, filepath2])

        embedder = self.image_encoder if is_image_mode else (self.text_encoder if is_text_mode else None)

        if embedder is None:
            raise ValueError("Unsupported file type: Both files must be of the same type e.g both image files or both text files")

        embeddings = embedder.embed_batch([filepath1, filepath2])
        return np.dot(embeddings[0], embeddings[1])
    

    def compare_embedding_to_dir(self, embedding: np.ndarray, dirpath: str, embedder: EmbeddingProvider, mode: AnalyserMode):
        prototype_embedding_filepath = self._get_prototype_path(dirpath, mode)
        if os.path.exists(prototype_embedding_filepath) and not self.is_prototype_stale(prototype_embedding_filepath):
            prototype_embedding = load_embedding(prototype_embedding_filepath)
        else:
            prototype_embedding = self.generate_prototype_for_dir(dirpath, embedder, mode)
            save_embedding(prototype_embedding_filepath, prototype_embedding)
        return np.dot(embedding, prototype_embedding)
    
    
    def compare_file_to_dir(self, filepath: str, dirpath: str):
        """
        Compute the cosine similarity between a file and a directory.
        """
        is_image_mode = self.are_files_valid(self.valid_img_exts, [filepath])
        is_text_mode = self.are_files_valid(self.valid_txt_exts, [filepath])

        if is_image_mode:
            mode = AnalyserMode.IMAGE
            embedder = self.image_encoder 
        elif is_text_mode:
            mode = AnalyserMode.TEXT
            embedder = self.text_encoder
        else:
            raise ValueError("Unsupported file type")

        file_embedding = embedder.embed(filepath)
        return self.compare_embedding_to_dir(file_embedding, dirpath, embedder, mode)


    def compare_file_to_dirs(self, filepath: str, dirpaths: List[str]) -> dict[str, float]:
            """
            Compute the cosine similarities between a file and multiple directories, and return
            a directory to similarity dict.
            """

            dirs_similarities: dict[str, float] = {}

            is_image_mode = self.are_files_valid(self.valid_img_exts, [filepath])
            is_text_mode = self.are_files_valid(self.valid_txt_exts, [filepath])

            if is_image_mode:
                mode = AnalyserMode.IMAGE
                embedder = self.image_encoder 
            elif is_text_mode:
                mode = AnalyserMode.TEXT
                embedder = self.text_encoder
            else:
                raise ValueError("Unsupported file type")
        
            file_embedding = embedder.embed(filepath)

            for dirpath in dirpaths:
                try:
                    similarity = self.compare_embedding_to_dir(file_embedding, dirpath, embedder, mode)
                except Exception as e:
                    print(f"[WARNING] Error comparing embeddings for directory {dirpath}: {e}")
                    continue
                
                dirs_similarities[dirpath] = similarity

            return dirs_similarities
    

    def generate_prototype_for_dir(self, dirpath, embedder: EmbeddingProvider, mode: AnalyserMode):
        files = get_files_from_dirs([dirpath], limit=self.max_files_for_prototypes)
        pos = 0
        chunk_size = 4
        embeddings = []
        while(pos < len(files)):
            file_batch = files[pos : (pos + chunk_size)]
            batch_embeddings = embedder.embed_batch(file_batch)
            embeddings.append(batch_embeddings)
            pos += chunk_size
        
        embeddings = np.array(batch_embeddings)
        prototype_embedding = generate_prototype_embedding(embeddings)
        return prototype_embedding
    

    def is_prototype_stale(self, path: str) -> bool:
        return get_days_since_last_modified(path) > self.refresh_prototype_duration
    
    def are_files_valid(self, allowed_exts: list[str], files: list[str]) -> bool:
        return all(path.lower().endswith(allowed_exts) for path in files)
    
    def _get_prototype_path(self, dirpath, mode: AnalyserMode):
        # This allows generating seperate prototypes for dirs which may have mutliple file types e.g Downloads
        if mode == AnalyserMode.IMAGE:
            prefix = "image"
        elif mode == AnalyserMode.TEXT:
            prefix = "text"
        elif mode == AnalyserMode.VIDEO:
            prefix = "video"

        return os.path.join(dirpath, f".{prefix}_prototype_embedding.pkl")