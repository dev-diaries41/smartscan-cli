from smartscan.processor import BatchProcessor, ProcessorListener
from smartscan.embeds import embed_video, embed_text, embed_image
from smartscan.providers import ImageEmbeddingProvider, TextEmbeddingProvider
from smartscan.types import StoredEmbedding


class ImageIndexer(BatchProcessor[str, StoredEmbedding]):
    def __init__(self, 
                image_encoder: ImageEmbeddingProvider, 
                listener = ProcessorListener[str, StoredEmbedding],
                **kwargs
                ):
        super().__init__(listener=listener, **kwargs)
        self.image_encoder = image_encoder

    def on_process(self, item):
        embedding = embed_image(self.image_encoder, item)
        return StoredEmbedding(item, embedding)
             
    # delegate to lister e.g to handle storage
    async def on_batch_complete(self, batch):
        await self.listener.on_batch_complete(batch)


        
class VideoIndexer(BatchProcessor[str, StoredEmbedding]):
    def __init__(self, 
                image_encoder: ImageEmbeddingProvider, 
                n_frames: int = 10,
                listener = ProcessorListener[str, StoredEmbedding],
                **kwargs
                ):
        super().__init__(listener=listener, **kwargs)
        self.image_encoder = image_encoder
        self.n_frames = n_frames

    def on_process(self, item):
        embedding = embed_video(self.image_encoder, item, self.n_frames)
        return StoredEmbedding(item, embedding)
             
    # delegate to lister e.g to handle storage
    async def on_batch_complete(self, batch):
        await self.listener.on_batch_complete(batch)


class DocIndexer(BatchProcessor[str, list[StoredEmbedding]]):
    def __init__(self, 
                text_encoder: TextEmbeddingProvider,
                listener = ProcessorListener[str, StoredEmbedding],
                max_chunks: int | None = None,
                **kwargs
                ):
        super().__init__(listener=listener, **kwargs)
        self.text_encoder = text_encoder
        self.max_chunks = max_chunks

    # All chunks share the same item_id (url or file) so that chunks are group
    # In the on_batch_complete method, the listener can handle use it as metaddata and assign unique ids to each chunk if required
    def on_process(self, item):
        chunk_embeddings = embed_text(self.text_encoder, item, self.text_encoder.max_tokenizer_length, self.max_chunks)
        return [StoredEmbedding(item, embedding) for embedding in chunk_embeddings]
             
    # delegate to lister e.g to handle storage
    async def on_batch_complete(self, batch):
        await self.listener.on_batch_complete(batch)