import numpy as np

from smartscan.processor import BatchProcessor, ProcessorListener
from smartscan.utils import are_valid_files
from smartscan.embeddings import embed_video_file, embed_text_file, embed_image_file
from smartscan.providers import ImageEmbeddingProvider, TextEmbeddingProvider
from smartscan.errors import SmartScanError, ErrorCode
from smartscan.constants import SupportedFileTypes


class FileIndexer(BatchProcessor[str, tuple[str, np.ndarray]]):
    def __init__(self, 
                image_encoder: ImageEmbeddingProvider, 
                text_encoder: TextEmbeddingProvider,
                n_frames: int = 10,
                n_chunks: int = 5,
                listener = ProcessorListener[str, tuple[str, np.ndarray]],
                **kwargs
                ):
        super().__init__(listener=listener, **kwargs)
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.n_frames = n_frames
        self.n_chunks = n_chunks
        self.valid_img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        self.valid_txt_exts = ('.txt', '.md', '.rst', '.html', '.json')
        self.valid_vid_exts = ('.mp4', '.mkv', '.webm')

    def on_process(self, item):
            file_embedding = self._embed_file(item)
            return item, file_embedding
             
    # delegate to lister e.g to handle storage
    async def on_batch_complete(self, batch):
        await self.listener.on_batch_complete(batch)


    def _embed_file(self, path: str) -> np.ndarray:
        is_image_file = are_valid_files(self.valid_img_exts, [path])
        is_text_file = are_valid_files(self.valid_txt_exts, [path])
        is_video_file = are_valid_files(self.valid_vid_exts, [path])

        if is_text_file:
            return embed_text_file(path, self.text_encoder, 128, self.n_chunks)
        elif is_image_file:
            return embed_image_file(path, self.image_encoder)
        elif is_video_file:
            return embed_video_file(path, self.n_frames, self.image_encoder)
        raise SmartScanError("Unsupported file type", code=ErrorCode.UNSUPPORTED_FILE_TYPE, details=f"Supported file types: {SupportedFileTypes.IMAGE + SupportedFileTypes.TEXT + SupportedFileTypes.VIDEO}")
    
        