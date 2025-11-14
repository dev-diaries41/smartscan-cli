from ml.providers.embeddings.embedding_provider import EmbeddingProvider
import os
from utils import save_embedding, load_embedding, has_one_week_passed
from typing import List, Callable, Any
import numpy as np
from utils import generate_prototype_embedding
from enum import IntEnum

class FileType(IntEnum):
    TEXT = 0
    IMAGE = 1
    VIDEO = 2

class FileOrganiser():
    def __init__(self, 
                 image_encoder: EmbeddingProvider, 
                 text_encoder: EmbeddingProvider,
                 similarity_threshold: float,
                 max_files_for_prototypes: int = 30,
                 ):
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.similarity_threshold = similarity_threshold
        self.valid_img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        self.valid_txt_exts = ('.txt', '.md', '.rst', '.html', '.json')
        self.max_files_for_prototypes = max_files_for_prototypes

    def compare_files(self, filepath1: str, filepath2: str):
        """Compute the cosine similarity between two files' embeddings."""
        is_image_mode = all(path.lower().endswith(self.valid_img_exts) for path in [filepath1, filepath2])
        is_text_mode = all(path.lower().endswith(self.valid_txt_exts) for path in [filepath1, filepath2])

        embedder = self.image_encoder if is_image_mode else (self.text_encoder if is_text_mode else None)

        assert embedder is not None, "Both files must be of the same type e.g both image files or both text files"

        embeddings = embedder.embed_batch([filepath1, filepath2])
        return np.dot(embeddings[0], embeddings[1])
    

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
    
    def compare_file_to_dir(self, filepath: str, dirpath: str):
        """
        Compute the cosine similarity between a text file's embedding and a directory's prototype embedding.
        """
        prototype_embedding_filepath = os.path.join(dirpath, "prototype_embedding.pkl")

        is_image_mode = all(path.lower().endswith(self.valid_img_exts) for path in [filepath])
        is_text_mode = all(path.lower().endswith(self.valid_txt_exts) for path in [filepath])

        embedder = self.image_encoder if is_image_mode else (self.text_encoder if is_text_mode else None)

        assert embedder is not None, "Both files must be of the same type e.g both image files or both text files"

    
        if os.path.exists(prototype_embedding_filepath):
            prototype_embedding = load_embedding(prototype_embedding_filepath)
        else:
            prototype_embedding = self.generate_prototype_for_dir(dirpath, embedder)
            save_embedding(prototype_embedding_filepath, prototype_embedding)

        file_embedding = embedder.embed(filepath)
        return np.dot(file_embedding, prototype_embedding)
    

    def compare_file_to_dirs(self, filepath: str, dirpaths: List[str]):
            """
            Compare a text file's embedding to prototype embeddings from multiple directories, returning
            the best matching directory and its similarity score.
            """
            best_similarity = -float("inf")
            best_dirpath = None

            is_image_mode = all(path.lower().endswith(self.valid_img_exts) for path in [filepath])
            is_text_mode = all(path.lower().endswith(self.valid_txt_exts) for path in [filepath])

            embedder = self.image_encoder if is_image_mode else (self.text_encoder if is_text_mode else None)

            assert embedder is not None, "Both files must be of the same type e.g both image files or both text files"

    
            file_embedding = embedder.embed(filepath)

            for dirpath in dirpaths:
                prototype_embedding_filepath = os.path.join(dirpath, "prototype_embedding.pkl")
                try:
                    if os.path.exists(prototype_embedding_filepath) and not has_one_week_passed(prototype_embedding_filepath):
                        prototype_embedding = load_embedding(prototype_embedding_filepath)
                    else:
                        prototype_embedding = self.generate_prototype_for_dir(dirpath, embedder)
                        save_embedding(prototype_embedding_filepath, prototype_embedding)
                except Exception as e:
                    print(f"[WARNING] Skipping directory {dirpath} due to error: {e}")
                    continue

                try:
                    similarity = np.dot(file_embedding, prototype_embedding)
                except Exception as e:
                    print(f"[WARNING] Error comparing embeddings for directory {dirpath}: {e}")
                    continue

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_dirpath = dirpath

            return best_dirpath, best_similarity

    def scan(self, target_dirs: List[str], destination_dirs: List[str], on_threshold_reached: Callable[[str, str], None]) -> None:
            """
            Scan through target directories, compare each file with destination directories,
            and invoke the on_threshold_reached callback when a similarity threshold is met.
            """
            for d in target_dirs:
                if os.path.isdir(d):
                    for filename in os.listdir(d):
                        file_path = os.path.join(d, filename)
                        if os.path.isfile(file_path):
                            best_dirpath, best_similarity = self.compare_file_to_dirs(file_path, destination_dirs)
                            
                            if best_similarity >= self.similarity_threshold:
                                try:
                                    on_threshold_reached(file_path, best_dirpath)
                                except Exception as e:
                                    print(f"[WARNING] Callback on_threshold_reached error: {e}")
                else:
                    print(f"[WARNING] Invalid directory: {d}")