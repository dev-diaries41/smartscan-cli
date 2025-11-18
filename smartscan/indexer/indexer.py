import numpy as np
from PIL import Image

from smartscan.processor.processor import BatchProcessor
from smartscan.processor.processor_listener import ProcessorListener
from smartscan.utils.file import read_text_file, are_valid_files
from smartscan.utils.embeddings import embed_video
from smartscan.ml.providers.embeddings.embedding_provider import ImageEmbeddingProvider, TextEmbeddingProvider


class FileIndexer(BatchProcessor[str, tuple[str, np.ndarray]]):
    def __init__(self, 
                image_encoder: ImageEmbeddingProvider, 
                text_encoder: TextEmbeddingProvider,
                n_frames: int = 10,
                listener = ProcessorListener[str, tuple[str, np.ndarray]],
                **kwargs
                ):
        super().__init__(listener=listener, **kwargs)
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.n_frames = n_frames
        self.valid_img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        self.valid_txt_exts = ('.txt', '.md', '.rst', '.html', '.json')
        self.valid_vid_exts = ('.mp4', '.mkv', '.webm')

    def on_process(self, item):
            filepath = item
            is_image_file = are_valid_files(self.valid_img_exts, [filepath])
            is_text_file = are_valid_files(self.valid_txt_exts, [filepath])
            is_video_file = are_valid_files(self.valid_vid_exts, [filepath])

            if is_image_file:
                embedder = self.image_encoder 
                file_embedding = embedder.embed(Image.open(filepath))
            elif is_text_file:
                embedder = self.text_encoder
                file_embedding = embedder.embed(read_text_file(filepath))
            elif is_video_file:
                embedder = self.image_encoder
                file_embedding = embed_video(filepath, self.n_frames, self.image_encoder)
            else:
                raise ValueError("Unsupported file type")
            
            return filepath, file_embedding
             
    
    async def on_batch_complete(self, batch):
        self.listener.on_batch_complete(batch)
        