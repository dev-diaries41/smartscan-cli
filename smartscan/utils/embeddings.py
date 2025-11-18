
import numpy as np
from PIL import Image

from smartscan.utils.file import get_frames_from_video


def generate_prototype_embedding(embeddings) -> np.ndarray:    
    embeddings_tensor = np.stack(embeddings, axis=0)
    prototype = np.mean(embeddings_tensor, axis=0)
    prototype /= np.linalg.norm(prototype)
    return prototype

def embed_video(path: str, n_frames: int, embedder):
    frame_arrs = get_frames_from_video(path, n_frames)
    frame_images = [Image.fromarray(frame) for frame in frame_arrs]
    batch = embedder.embed_batch(frame_images)
    return generate_prototype_embedding(batch)

def few_shot_classification(item_embedding:  np.ndarray, class_prototypes: list[tuple[str, np.ndarray]]) -> tuple[str, float]:
        class_similarities_dict: dict[str, float] = {}

        for class_id, prototype_embedding in class_prototypes:
            try:
                similarity = np.dot(item_embedding, prototype_embedding)
            except Exception as e:
                continue
            class_similarities_dict[class_id] = similarity

        class_similarities = sorted(class_similarities_dict.items(), key=lambda x: x[1], reverse=True)        
        return class_similarities[0]


def chunk_text(s: str, tokenizer_max_length: int, limit: int = 10):
    max_chunks = len(s) // 4 * tokenizer_max_length
    n_chunks = min(limit, max_chunks)
    chunks = []
    start = 0

    while len(chunks) < n_chunks:
        end = start + tokenizer_max_length
        if end >= len(s):
            chunk = s[start:]
        else:
            space_index = s.rfind(" ", start, end)
            if space_index == -1: 
                space_index = end
            chunk = s[start:space_index]
            end = space_index
        if not chunk:
            break
        chunks.append(chunk)
        start = end + 1

    return chunks
