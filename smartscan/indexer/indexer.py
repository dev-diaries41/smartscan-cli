import numpy as np
from PIL import Image

from smartscan.processor.processor import BatchProcessor
from smartscan.ml.providers.embeddings.embedding_provider import ImageEmbeddingProvider, TextEmbeddingProvider
from smartscan.utils.file import read_text_file
from smartscan.utils.embeddings import embed_video
from chromadb import Collection

class FileIndexer(BatchProcessor[str, tuple[str, np.ndarray]]):
    def __init__(self, 
                image_encoder: ImageEmbeddingProvider, 
                text_encoder: TextEmbeddingProvider,
                store: Collection,
                n_frames: int = 10,
                **kwargs
                ):
        super().__init__(**kwargs)
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.store = store
        self.n_frames = n_frames
        self.valid_img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        self.valid_txt_exts = ('.txt', '.md', '.rst', '.html', '.json')
        self.valid_vid_exts = ('.mp4', '.mkv', '.webm')

    def on_process(self, item):
            filepath = item
            is_image_mode = self._are_files_valid(self.valid_img_exts, [filepath])
            is_text_mode = self._are_files_valid(self.valid_txt_exts, [filepath])
            is_video_mode = self._are_files_valid(self.valid_vid_exts, [filepath])

            if is_image_mode:
                embedder = self.image_encoder 
                file_embedding = embedder.embed(Image.open(filepath))
            elif is_text_mode:
                embedder = self.text_encoder
                file_embedding = embedder.embed(read_text_file(filepath))
            elif is_video_mode:
                embedder = self.image_encoder
                file_embedding = embed_video(filepath, self.n_frames, self.image_encoder)
            else:
                raise ValueError("Unsupported file type")
            
            return filepath, file_embedding
             
    
    async def on_batch_complete(self, batch):
        if len(batch) <= 0:
            return
        ids, embeddings = map(list, zip(*batch))
        self.store.add(
            ids = ids,
            embeddings=embeddings
        )
       
    
    def _are_files_valid(self, allowed_exts: list[str], files: list[str]) -> bool:
        return all(path.lower().endswith(allowed_exts) for path in files)
     