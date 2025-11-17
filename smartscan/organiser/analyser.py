import os
import numpy as np
import pickle
from typing import List
from enum import IntEnum
from PIL import Image
from chromadb import Collection

from smartscan.ml.providers.embeddings.embedding_provider import ImageEmbeddingProvider, TextEmbeddingProvider, EmbeddingProvider
from smartscan.utils.file import get_days_since_last_modified, get_files_from_dirs, read_text_file
from smartscan.utils.embeddings import generate_prototype_embedding, embed_video, chunk_text


class AnalyserMode(IntEnum):
    TEXT = 0
    IMAGE = 1
    VIDEO = 2

class FileAnalyser:
    def __init__(self, 
                image_encoder: ImageEmbeddingProvider, 
                text_encoder: TextEmbeddingProvider,
                text_store: Collection,
                image_store: Collection,
                video_store: Collection,
                similarity_threshold: float,
                max_files_for_prototypes: int = 30,
                refresh_prototype_duration: int  = 7,
                n_frames_limit = 10,
                n_chunks_limit = 5
                 ):
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.text_store = text_store
        self.image_store = image_store
        self.video_store = video_store
        self.similarity_threshold = similarity_threshold
        self.valid_img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        self.valid_txt_exts = ('.txt', '.md', '.rst', '.html', '.json')
        self.valid_vid_exts = ('.mp4', '.mkv', '.webm')
        self.max_files_for_prototypes = max_files_for_prototypes
        self.refresh_prototype_duration = refresh_prototype_duration
        self.n_frames = n_frames_limit
        self.n_chunks = n_chunks_limit
        
    def compare_files(self, files: list[str]):
        """Compute the cosine similarity between two files"""
        mode = self._get_mode(files)
        embeddings = self._get_or_embed_batch(files, mode)
        return np.dot(embeddings[0], embeddings[1])
    

    def compare_embedding_to_dir(self, embedding: np.ndarray, dirpath: str, mode: AnalyserMode):
        prototype_embedding_filepath = self._get_prototype_path(dirpath, mode)
        if os.path.exists(prototype_embedding_filepath) and not self._is_prototype_stale(prototype_embedding_filepath):
            prototype_embedding = self.load_embedding(prototype_embedding_filepath)
        else:
            prototype_embedding = self._generate_prototype_for_dir(dirpath, mode)
        self.save_embedding(prototype_embedding_filepath, prototype_embedding)
        return np.dot(embedding, prototype_embedding)
    
    
    def compare_file_to_dir(self, filepath: str, dirpath: str):
        """
        Compute the cosine similarity between a file and a directory.
        """
        mode = self._get_mode([filepath])
        file_embedding = self._get_or_embed(filepath, mode)
        return self.compare_embedding_to_dir(file_embedding, dirpath, mode)

    def compare_file_to_dirs(self, filepath: str, dirpaths: List[str]) -> dict[str, float]:
            """
            Compute the cosine similarities between a file and multiple directories, and return
            a directory to similarity dict.
            """

            dirs_similarities: dict[str, float] = {}
            mode = self._get_mode([filepath])
            file_embedding = self._get_or_embed(filepath, mode)

            for dirpath in dirpaths:
                try:
                    similarity = self.compare_embedding_to_dir(file_embedding, dirpath, mode)
                except Exception as e:
                    print(f"[WARNING] Error comparing embeddings for directory {dirpath}: {e}")
                    continue
                dirs_similarities[dirpath] = similarity

            return dirs_similarities
    
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

    def _generate_prototype_for_dir(self, dirpath, mode: AnalyserMode):
        if mode == AnalyserMode.IMAGE:
            allowed_exts = self.valid_img_exts
        elif mode == AnalyserMode.TEXT:
            allowed_exts = self.valid_txt_exts
        elif mode == AnalyserMode.VIDEO:
            allowed_exts = self.valid_vid_exts

        files = get_files_from_dirs([dirpath], dir_skip_patterns=["**/.*"], limit=self.max_files_for_prototypes, allowed_exts=allowed_exts)

        if not files:
            raise ValueError(f"Error generating prototype: no files found for directory: {dirpath}")
        
        pos = 0
        chunk_size = 4
        embeddings = []
        while(pos < len(files)):
            file_batch = files[pos : (pos + chunk_size)]
            batch_embeddings = self._get_or_embed_batch(file_batch, mode)
            embeddings.append(batch_embeddings)
            pos += chunk_size
        
        embeddings = np.array(batch_embeddings)
        prototype_embedding = generate_prototype_embedding(embeddings)
        return prototype_embedding
    

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
    
    def _get_mode(self, files: str) -> AnalyserMode:
        is_image_mode = self._are_files_valid(self.valid_img_exts, files)
        is_text_mode = self._are_files_valid(self.valid_txt_exts, files)
        is_video_mode = self._are_files_valid(self.valid_vid_exts, files)

        if is_image_mode:
            mode = AnalyserMode.IMAGE
        elif is_text_mode:
            mode = AnalyserMode.TEXT
        elif is_video_mode:
            mode = AnalyserMode.VIDEO
        else:
            raise ValueError("Unsupported file type")
        return mode
    
    def _get_or_embed(self, filepath: str, mode: AnalyserMode):
        if mode == AnalyserMode.IMAGE:
            embedder = self.image_encoder 
            result = self.image_store.get(ids=[filepath], include=['embeddings']) 
            if len(result['embeddings']) == 1:
                file_embedding = result['embeddings']
            else:
                file_embedding = embedder.embed(Image.open(filepath))
        elif mode == AnalyserMode.TEXT:
            embedder = self.text_encoder
            result = self.text_store.get(ids=[filepath], include=['embeddings']) 
            if len(result['embeddings']) == 1:
                file_embedding = result['embeddings']
            else:
                chunks = chunk_text(read_text_file(filepath), 128, self.n_chunks)
                chunk_embeddings = embedder.embed_batch(chunks)
                file_embedding = generate_prototype_embedding(chunk_embeddings)
        elif mode == AnalyserMode.VIDEO:
            embedder = self.image_encoder
            result = self.video_store.get(ids=[filepath], include=['embeddings']) 
            if len(result['embeddings']) == 1:
                file_embedding = result['embeddings']
            else:
                file_embedding = embed_video(filepath, self.n_frames, self.image_encoder)
        else:
            raise ValueError("Unsupported file type")
        
        return file_embedding
    

    def _get_or_embed_batch(self, files: list[str], mode: AnalyserMode):
        if mode == AnalyserMode.IMAGE:
            embedder = self.image_encoder
            result = self.image_store.get(ids=files, include=['embeddings']) 
            if len(result['embeddings']) == len(files):
                embeddings = result['embeddings']
            else:
                embeddings = embedder.embed_batch([Image.open(path) for path in files])
        elif mode == AnalyserMode.TEXT:
            embedder = self.text_encoder
            result = self.text_store.get(ids=files, include=['embeddings']) 
            if len(result['embeddings']) == len(files):
                embeddings = result['embeddings']
            else:
                text_batch = [read_text_file(path) for path in files]
                chunks_batch = [chunk_text(text, 128, self.n_chunks) for text in text_batch]
                embeddings_batch = [embedder.embed_batch(chunks) for chunks in chunks_batch]
                embeddings = np.stack([generate_prototype_embedding(embeds) for embeds in embeddings_batch], axis=0)
        elif mode == AnalyserMode.VIDEO:
            result = self.video_store.get(ids=files, include=['embeddings']) 
            if len(result['embeddings']) == len(files):
                embeddings = result['embeddings']
            else:
                embeddings = np.stack([embed_video(path, self.n_frames, self.image_encoder) for path in files], axis=0)
        else:
            raise ValueError("Unsupported file type")
        
        return embeddings