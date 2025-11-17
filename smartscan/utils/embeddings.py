
import numpy as np
from PIL import Image

from smartscan.utils.file import get_frames_from_video
from smartscan.ml.providers.embeddings.minilm.text import MiniLmTextEmbedder
from smartscan.ml.providers.embeddings.dino.image import DinoSmallV2ImageEmbedder
from smartscan.ml.providers.embeddings.clip.text import ClipTextEmbedder
from smartscan.ml.providers.embeddings.clip.image import ClipImageEmbedder
from smartscan.ml.providers.embeddings.embedding_provider import ImageEmbeddingProvider, TextEmbeddingProvider
from smartscan.constants import CLIP_IMAGE_MODEL_PATH, DINO_V2_SMALL_MODEL_PATH, CLIP_TEXT_MODEL_PATH, MINILM_MODEL_PATH


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


def get_image_encoder(path: str) -> ImageEmbeddingProvider:
    if path == DINO_V2_SMALL_MODEL_PATH:
        return DinoSmallV2ImageEmbedder(path)
    elif path == CLIP_IMAGE_MODEL_PATH:
        return ClipImageEmbedder(path)
    raise ValueError(f"Invalid model path: {path}")

def get_text_encoder(path: str) -> TextEmbeddingProvider:
    if path == MINILM_MODEL_PATH:
        return MiniLmTextEmbedder(path)
    elif path == CLIP_TEXT_MODEL_PATH:
        return ClipTextEmbedder(path)
    raise ValueError(f"Invalid model path: {path}")

    





