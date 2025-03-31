import os
from utils import save_embedding, load_embedding
from typing import List, Callable, Any
from embeddings import (get_embedding, get_embedding_onnx, calculate_cosine_similarity, aggregate_embeddings_from_dir)

def compare_files(filepath1: str, filepath2: str, tokenizer, model = None):
    """Compute the cosine similarity between two text files' embeddings."""

    if model is None:
        file_embedding1 = get_embedding_onnx(tokenizer, filepath1)
        file_embedding2 = get_embedding_onnx(tokenizer, filepath2)
    else:
        file_embedding1 = get_embedding(model, tokenizer, filepath1)
        file_embedding2 = get_embedding(model, tokenizer, filepath2)

    return calculate_cosine_similarity(file_embedding1, file_embedding2)


def compare_file_to_dir(filepath: str, dirpath: str, tokenizer, model = None):
    """
    Compute the cosine similarity between a text file's embedding and a directory's prototype embedding.
    """
    prototype_embedding_filepath = os.path.join(dirpath, "prototype_embedding.pkl")
  
    if os.path.exists(prototype_embedding_filepath):
        prototype_embedding = load_embedding(prototype_embedding_filepath)
    else:
        prototype_embedding = aggregate_embeddings_from_dir(tokenizer, dirpath, model)
        save_embedding(prototype_embedding_filepath, prototype_embedding)

    if model is None:
        file_embedding = get_embedding_onnx(tokenizer, filepath)
    else:
        file_embedding = get_embedding(model, tokenizer, filepath )

    return calculate_cosine_similarity(file_embedding, prototype_embedding)


def compare_file_to_dirs(filepath: str, dirpaths: List[str], tokenizer, model = None):
    """
    Compare a text file's embedding to prototype embeddings from multiple directories, returning
    the best matching directory and its similarity score.
    """
    best_similarity = -float("inf")
    best_dirpath = None
  
    if model is None:
        file_embedding = get_embedding_onnx(tokenizer, filepath)
    else:
        file_embedding = get_embedding(model, tokenizer, filepath )
  
    for dirpath in dirpaths:
        prototype_embedding_filepath = os.path.join(dirpath, "prototype_embedding.pkl")
        try:
            if os.path.exists(prototype_embedding_filepath):
                prototype_embedding = load_embedding(prototype_embedding_filepath)
            else:
                prototype_embedding = aggregate_embeddings_from_dir(tokenizer, dirpath, model)
                save_embedding(prototype_embedding_filepath, prototype_embedding)
        except Exception as e:
            print(f"[WARNING] Skipping directory {dirpath} due to error: {e}")
            continue

        try:
            similarity = calculate_cosine_similarity(file_embedding, prototype_embedding)
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
