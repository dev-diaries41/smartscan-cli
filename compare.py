import os
from utils import save_embedding, load_embedding, has_one_week_passed
from typing import List, Callable, Any
from sdk.models.providers.embeddings.embedding_provider import EmbeddingProvider
import numpy as np
from utils import generate_prototype_embedding



def compare_files(embedder: EmbeddingProvider, filepath1: str, filepath2: str):
    """Compute the cosine similarity between two files' embeddings."""

    embeddings = embedder.embed_batch([filepath1, filepath2])
    return np.dot(embeddings[0], embeddings[1])


def generate_prototype_for_dir(dirpath, embedder: EmbeddingProvider, limit = 30):
    files = [os.path.join(dirpath, f) for f in os.listdir()[:limit]]
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


def compare_file_to_dir(filepath: str, dirpath: str, embedder: EmbeddingProvider, limit = 30):
    """
    Compute the cosine similarity between a text file's embedding and a directory's prototype embedding.
    """
    prototype_embedding_filepath = os.path.join(dirpath, "prototype_embedding.pkl")
  
    if os.path.exists(prototype_embedding_filepath):
        prototype_embedding = load_embedding(prototype_embedding_filepath)
    else:
        prototype_embedding = generate_prototype_for_dir(dirpath, embedder, limit)
        save_embedding(prototype_embedding_filepath, prototype_embedding)

    file_embedding = embedder.embed(filepath)
    return np.dot(file_embedding, prototype_embedding)


def compare_file_to_dirs(filepath: str, dirpaths: List[str],  embedder: EmbeddingProvider, limit = 30):
    """
    Compare a text file's embedding to prototype embeddings from multiple directories, returning
    the best matching directory and its similarity score.
    """
    best_similarity = -float("inf")
    best_dirpath = None
  
    file_embedding = embedder.embed(filepath)

    for dirpath in dirpaths:
        prototype_embedding_filepath = os.path.join(dirpath, "prototype_embedding.pkl")
        try:
            if os.path.exists(prototype_embedding_filepath) and not has_one_week_passed(prototype_embedding_filepath):
                prototype_embedding = load_embedding(prototype_embedding_filepath)
            else:
                prototype_embedding = generate_prototype_for_dir(dirpath, embedder, limit)
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

def scan(target_dirs: List[str], destination_dirs: List[str], tokenizer, on_threshold_reached: Callable[[str, str], None], model = None, threshold: float = 0.7) -> None:
    """
    Scan through target directories, compare each file with destination directories,
    and invoke the on_threshold_reached callback when a similarity threshold is met.
    """
    for d in target_dirs:
        if os.path.isdir(d):
            for filename in os.listdir(d):
                file_path = os.path.join(d, filename)
                if os.path.isfile(file_path):
                    best_dirpath, best_similarity = compare_file_to_dirs(file_path, destination_dirs, tokenizer, model)
                    
                    if best_similarity >= threshold:
                        try:
                            on_threshold_reached(file_path, best_dirpath)
                        except Exception as e:
                            print(f"[WARNING] Callback on_threshold_reached error: {e}")
        else:
            print(f"[WARNING] Invalid directory: {d}")
