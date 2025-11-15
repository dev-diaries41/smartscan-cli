
import numpy as np
from smartscan.utils.file import get_frames_from_video
from PIL import Image

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


