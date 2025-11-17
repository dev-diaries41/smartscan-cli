import numpy as np
from PIL import Image
from chromadb import Collection

from smartscan.processor.processor import BatchProcessor
from smartscan.ml.providers.embeddings.embedding_provider import ImageEmbeddingProvider, TextEmbeddingProvider
from smartscan.ml.providers.embeddings.minilm.text import MiniLmTextEmbedder
from smartscan.ml.providers.embeddings.dino.image import DinoSmallV2ImageEmbedder
from smartscan.ml.providers.embeddings.clip.text import ClipTextEmbedder
from smartscan.ml.providers.embeddings.clip.image import ClipImageEmbedder
from smartscan.utils.file import read_text_file
from smartscan.utils.embeddings import embed_video
from smartscan.constants import CLIP_IMAGE_MODEL_PATH, DINO_V2_SMALL_MODEL_PATH, CLIP_TEXT_MODEL_PATH, MINILM_MODEL_PATH


class FileIndexer(BatchProcessor[str, tuple[str, np.ndarray]]):
    def __init__(self, 
                image_encoder_path: str, 
                text_encoder_path: str,
                text_store: Collection,
                image_store: Collection,
                video_store: Collection,
                n_frames: int = 10,
                **kwargs
                ):
        super().__init__(**kwargs)
        self.image_encoder = self._get_image_encoder(image_encoder_path)
        self.text_encoder = self._get_text_encoder(text_encoder_path)
        self.text_store = text_store
        self.image_store = image_store
        self.video_store = video_store
        self.n_frames = n_frames
        self.valid_img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        self.valid_txt_exts = ('.txt', '.md', '.rst', '.html', '.json')
        self.valid_vid_exts = ('.mp4', '.mkv', '.webm')

    def on_process(self, item):
            filepath = item
            is_image_file = self._are_files_valid(self.valid_img_exts, [filepath])
            is_text_file = self._are_files_valid(self.valid_txt_exts, [filepath])
            is_video_file = self._are_files_valid(self.valid_vid_exts, [filepath])

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
        if len(batch) <= 0:
            return
        partitions = { "image": ([], []), "text": ([], []), "video": ([], [])}

        for id_, embed in batch:
            is_image_file = self._are_files_valid(self.valid_img_exts, [id_])
            is_text_file = self._are_files_valid(self.valid_txt_exts, [id_])
            is_video_file = self._are_files_valid(self.valid_vid_exts, [id_])

            if is_image_file:
                partitions['image'][0].append(id_)
                partitions['image'][1].append(embed)
            elif is_text_file:
                partitions['text'][0].append(id_)
                partitions['text'][1].append(embed)
            elif is_video_file:
                partitions['video'][0].append(id_)
                partitions['video'][1].append(embed)
        
        if len(partitions['image'][0]) > 0:
            self.image_store.add(ids = partitions['image'][0],embeddings=partitions['image'][1])
        if len(partitions['text'][0]) > 0:
            self.text_store.add(ids = partitions['text'][0],embeddings=partitions['text'][1])
        if len(partitions['video'][0]) > 0:
            self.video_store.add(ids = partitions['video'][0],embeddings=partitions['video'][1])
                
    def filter(self, items: list[str]) -> list[str]:
        image_ids = self._get_exisiting_ids(self.image_store)
        text_ids = self._get_exisiting_ids(self.text_store)
        video_ids = self._get_exisiting_ids(self.video_store)
        exclude = set(image_ids) | set(text_ids) | set(video_ids)
        return [item for item in items if item not in exclude]
       
    def _are_files_valid(self, allowed_exts: list[str], files: list[str]) -> bool:
        return all(path.lower().endswith(allowed_exts) for path in files)

    def _get_exisiting_ids(self, store: Collection) -> list:
        limit = 100
        offset = 0
        ids = []

        while True:
            batch = store.get(limit=limit, offset=offset)
            if not batch['ids']:
                break
            ids.extend(batch['ids'])
            offset += limit
        return ids
    

    @staticmethod
    def _get_image_encoder(path: str) -> ImageEmbeddingProvider:
        if path == DINO_V2_SMALL_MODEL_PATH:
            return DinoSmallV2ImageEmbedder(path)
        elif path == CLIP_IMAGE_MODEL_PATH:
            return ClipImageEmbedder(path)
        raise ValueError(f"Invalid model path: {path}")
    
    @staticmethod
    def _get_text_encoder(path: str) -> TextEmbeddingProvider:
        if path == MINILM_MODEL_PATH:
            return MiniLmTextEmbedder(path)
        elif path == CLIP_TEXT_MODEL_PATH:
            return ClipTextEmbedder(path)
        raise ValueError(f"Invalid model path: {path}")