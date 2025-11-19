import numpy as np
from dataclasses import dataclass

from smartscan.processor import BatchProcessor, ProcessorListener
from smartscan.providers import ImageEmbeddingProvider, TextEmbeddingProvider
from smartscan.embeddings import few_shot_classification, embed_image_file, embed_text_file, embed_video_file
from smartscan.utils import are_valid_files
from smartscan.errors import SmartScanError, ErrorCode
from smartscan.constants import SupportedFileTypes


@dataclass
class ClassificationResult:
    item: str
    class_id: str
    similarity: float


class FileClassifier(BatchProcessor[str, ClassificationResult]):
    def __init__(self, 
                image_encoder: ImageEmbeddingProvider, 
                text_encoder: TextEmbeddingProvider,                 
                class_prototypes: list[tuple[str, np.ndarray]],
                listener: ProcessorListener[str, ClassificationResult],
                similarity_threshold: float,
                n_frames_limit = 10,
                n_chunks_limit = 5,
                **kwargs
                ):
        super().__init__(listener=listener, **kwargs)
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder        
        self.class_prototypes = class_prototypes
        self.similarity_threshold = similarity_threshold
        self.n_frames = n_frames_limit
        self.n_chunks = n_chunks_limit

    def on_process(self, item):
        file_embedding = self._embed_file(item)
        destination_dir, best_similarity = few_shot_classification(file_embedding, self.class_prototypes)
        
        if best_similarity <= self.similarity_threshold:
            raise SmartScanError("Item unclassified", ErrorCode.BELOW_SIMILARITY_THRESHOLD)

        return ClassificationResult(item, destination_dir, best_similarity)
    
    
    async def on_batch_complete(self, batch):
        self.listener.on_batch_complete(batch)

    
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
    