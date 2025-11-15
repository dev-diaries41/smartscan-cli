import os
import numpy as np
import pickle
from typing import List
from enum import IntEnum
from PIL import Image

from smartscan.ml.providers.embeddings.embedding_provider import EmbeddingProvider
from smartscan.utils.file import get_days_since_last_modified, get_files_from_dirs, read_text_file
from smartscan.utils.embeddings import generate_prototype_embedding, embed_video


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
                 n_frames = 10
                 ):
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.similarity_threshold = similarity_threshold
        self.valid_img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        self.valid_txt_exts = ('.txt', '.md', '.rst', '.html', '.json')
        self.valid_vid_exts = ('.mp4', '.mkv', '.webm')
        self.max_files_for_prototypes = max_files_for_prototypes
        self.refresh_prototype_duration = refresh_prototype_duration
        self.n_frames = n_frames
        

    def compare_files(self, filepath1: str, filepath2: str):
        """Compute the cosine similarity between two files"""
        is_image_mode = self._are_files_valid(self.valid_img_exts, [filepath1, filepath2])
        is_text_mode = self._are_files_valid(self.valid_txt_exts, [filepath1, filepath2])
        is_video_mode = self._are_files_valid(self.valid_vid_exts, [filepath1, filepath2])

        if is_image_mode:
            embedder = self.image_encoder 
            embeddings = embedder.embed_batch([Image.open(path) for path in [filepath1, filepath2]])
        elif is_text_mode:
            embedder = self.text_encoder
            embeddings = embedder.embed_batch([read_text_file(path) for path in [filepath1, filepath2]])
        elif is_video_mode:
            embeddings = np.stack([embed_video(path, self.n_frames, self.image_encoder) for path in [filepath1, filepath2]], axis=0)
        else:
            raise ValueError("Unsupported file type: Both files must be of the same type")

        return np.dot(embeddings[0], embeddings[1])
    

    def compare_embedding_to_dir(self, embedding: np.ndarray, dirpath: str, embedder: EmbeddingProvider, mode: AnalyserMode):
        prototype_embedding_filepath = self._get_prototype_path(dirpath, mode)
        if os.path.exists(prototype_embedding_filepath) and not self._is_prototype_stale(prototype_embedding_filepath):
            prototype_embedding = self.load_embedding(prototype_embedding_filepath)
        else:
            prototype_embedding = self._generate_prototype_for_dir(dirpath, embedder, mode)
            self.save_embedding(prototype_embedding_filepath, prototype_embedding)
        return np.dot(embedding, prototype_embedding)
    
    
    def compare_file_to_dir(self, filepath: str, dirpath: str):
        """
        Compute the cosine similarity between a file and a directory.
        """
        is_image_mode = self._are_files_valid(self.valid_img_exts, [filepath])
        is_text_mode = self._are_files_valid(self.valid_txt_exts, [filepath])
        is_video_mode = self._are_files_valid(self.valid_vid_exts, [filepath])

        if is_image_mode:
            mode = AnalyserMode.IMAGE
            embedder = self.image_encoder 
            file_embedding = embedder.embed(Image.open(filepath))
        elif is_text_mode:
            mode = AnalyserMode.TEXT
            embedder = self.text_encoder
            file_embedding = embedder.embed(read_text_file(filepath))
        elif is_video_mode:
            mode = AnalyserMode.VIDEO
            embedder = self.image_encoder
            file_embedding = embed_video(filepath, self.n_frames, self.image_encoder)
        else:
            raise ValueError("Unsupported file type")
        
        return self.compare_embedding_to_dir(file_embedding, dirpath, embedder, mode)


    def compare_file_to_dirs(self, filepath: str, dirpaths: List[str]) -> dict[str, float]:
            """
            Compute the cosine similarities between a file and multiple directories, and return
            a directory to similarity dict.
            """

            dirs_similarities: dict[str, float] = {}

            is_image_mode = self._are_files_valid(self.valid_img_exts, [filepath])
            is_text_mode = self._are_files_valid(self.valid_txt_exts, [filepath])
            is_video_mode = self._are_files_valid(self.valid_vid_exts, [filepath])

            if is_image_mode:
                mode = AnalyserMode.IMAGE
                embedder = self.image_encoder 
                file_embedding = embedder.embed(Image.open(filepath))
            elif is_text_mode:
                mode = AnalyserMode.TEXT
                embedder = self.text_encoder
                file_embedding = embedder.embed(read_text_file(filepath))
            elif is_video_mode:
                mode = AnalyserMode.VIDEO
                embedder = self.image_encoder
                file_embedding = embed_video(filepath, self.n_frames, self.image_encoder)
            else:
                raise ValueError("Unsupported file type")
        

            for dirpath in dirpaths:
                try:
                    similarity = self.compare_embedding_to_dir(file_embedding, dirpath, embedder, mode)
                except Exception as e:
                    print(f"[WARNING] Error comparing embeddings for directory {dirpath}: {e}")
                    continue
                
                dirs_similarities[dirpath] = similarity

            return dirs_similarities
    

    def _generate_prototype_for_dir(self, dirpath, embedder: EmbeddingProvider, mode: AnalyserMode):
        if mode == AnalyserMode.IMAGE:
            allowed_exts = self.valid_img_exts
        elif mode == AnalyserMode.TEXT:
            allowed_exts = self.valid_txt_exts
        elif mode == AnalyserMode.VIDEO:
            allowed_exts = self.valid_vid_exts

        files = get_files_from_dirs([dirpath], dir_skip_patterns=["**/.*"], limit=self.max_files_for_prototypes, allowed_exts=allowed_exts)
        pos = 0
        chunk_size = 4
        embeddings = []
        while(pos < len(files)):
            file_batch = files[pos : (pos + chunk_size)]
            if mode == AnalyserMode.IMAGE:
                batch_embeddings = embedder.embed_batch([Image.open(path) for path in file_batch])
            elif mode == AnalyserMode.TEXT:
                batch_embeddings = embedder.embed_batch([read_text_file(path) for path in file_batch])
            elif mode == AnalyserMode.VIDEO:
                batch_embeddings = np.stack([embed_video(path, self.n_frames, self.image_encoder) for path in file_batch], axis=0)

            embeddings.append(batch_embeddings)
            pos += chunk_size
        
        embeddings = np.array(batch_embeddings)
        prototype_embedding = generate_prototype_embedding(embeddings)
        return prototype_embedding
    
    @staticmethod
    def save_embedding(filepath: str, embedding: np.ndarray):
        """Saves embedding to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(embedding, f)

    @staticmethod
    def load_embedding(filepath: str) -> np.ndarray:
        """Loads embedding from a file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def _is_prototype_stale(self, path: str) -> bool:
        return get_days_since_last_modified(path) > self.refresh_prototype_duration
    
    def _are_files_valid(self, allowed_exts: list[str], files: list[str]) -> bool:
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
    
 