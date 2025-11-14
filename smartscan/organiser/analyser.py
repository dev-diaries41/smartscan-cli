from ml.providers.embeddings.embedding_provider import EmbeddingProvider
import os
from utils.file import save_embedding, load_embedding, get_days_since_last_modified
from utils.ml_ops import generate_prototype_embedding
from typing import List
import numpy as np


class FileAnalyser():
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
        self.max_files_for_prototypes = max_files_for_prototypes
        self.refresh_prototype_duration = refresh_prototype_duration

    def compare_files(self, filepath1: str, filepath2: str):
        """Compute the cosine similarity between two files"""
        is_image_mode = all(path.lower().endswith(self.valid_img_exts) for path in [filepath1, filepath2])
        is_text_mode = all(path.lower().endswith(self.valid_txt_exts) for path in [filepath1, filepath2])

        embedder = self.image_encoder if is_image_mode else (self.text_encoder if is_text_mode else None)

        if embedder is None:
            raise ValueError("Both files must be of the same type e.g both image files or both text files")

        embeddings = embedder.embed_batch([filepath1, filepath2])
        return np.dot(embeddings[0], embeddings[1])
    

    def compare_embedding_to_dir(self, embedding: np.ndarray, dirpath: str, embedder: EmbeddingProvider):
        prototype_embedding_filepath = os.path.join(dirpath, "prototype_embedding.pkl")
        if os.path.exists(prototype_embedding_filepath) and not self.is_prototype_stale(prototype_embedding_filepath):
            prototype_embedding = load_embedding(prototype_embedding_filepath)
        else:
            prototype_embedding = self.generate_prototype_for_dir(dirpath, embedder)
            save_embedding(prototype_embedding_filepath, prototype_embedding)
        return np.dot(embedding, prototype_embedding)
    
    
    def compare_file_to_dir(self, filepath: str, dirpath: str):
        """
        Compute the cosine similarity between a file and a directory.
        """
        is_image_mode = all(path.lower().endswith(self.valid_img_exts) for path in [filepath])
        is_text_mode = all(path.lower().endswith(self.valid_txt_exts) for path in [filepath])

        embedder = self.image_encoder if is_image_mode else (self.text_encoder if is_text_mode else None)

        if embedder is None:
            raise ValueError("Unsupported file type")

        file_embedding = embedder.embed(filepath)
        return self.compare_embedding_to_dir(file_embedding, dirpath, embedder)


    def compare_file_to_dirs(self, filepath: str, dirpaths: List[str]):
            """
            Compute the cosine similarities between a file and multiple directories, returning
            the best matching directory and its similarity score.
            """
            best_similarity = -float("inf")
            best_dirpath = None

            is_image_mode = all(path.lower().endswith(self.valid_img_exts) for path in [filepath])
            is_text_mode = all(path.lower().endswith(self.valid_txt_exts) for path in [filepath])

            embedder = self.image_encoder if is_image_mode else (self.text_encoder if is_text_mode else None)

            if embedder is None:
                raise ValueError("Unsupported file type")
    
            file_embedding = embedder.embed(filepath)

            for dirpath in dirpaths:
                try:
                    similarity = self.compare_embedding_to_dir(file_embedding, dirpath, embedder)
                except Exception as e:
                    print(f"[WARNING] Error comparing embeddings for directory {dirpath}: {e}")
                    continue

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_dirpath = dirpath

            return best_dirpath, best_similarity
    

    def generate_prototype_for_dir(self, dirpath, embedder: EmbeddingProvider):
        files = [os.path.join(dirpath, f) for f in os.listdir()[:self.max_files_for_prototypes]]
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