from sdk.models.providers.embeddings.embedding_provider import EmbeddingProvider
import torch
from sdk.models.onnx_model import OnnxModel
from sdk.models.providers.embeddings.clip.tokenizer.tokenizer import load_clip_tokenizer
import numpy as np

VOCAB_PATH = "sdk/models/providers/embeddings/clip/tokenizer/vocab.json"
MERGES_PATH = "sdk/models/providers/embeddings/clip/tokenizer/merges.txt"

class ClipTextEmbedder(EmbeddingProvider):
    def __init__(self, model_path: str):
        self._model = OnnxModel(model_path)
        self._embedding_dim = 512
        self._max_len = 77
        self.tokenizer = load_clip_tokenizer(VOCAB_PATH, MERGES_PATH)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, data: str):
        """Create vector embeddings for text using an ONNX model."""

        assert self._model.is_load(), "Model not loaded"
        
        input_name = self._model.get_inputs()[0].name
        token_ids = self._tokenize(data)
        token_input = np.array([token_ids], dtype=np.int64)
        outputs = self._model.run({input_name: token_input})
        embedding = outputs[0][0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    

    def embed_batch(self, data: list[str]):
        """Create vector embeddings for batch of text files using an ONNX model."""

        assert self._model.is_load(), "Model not loaded"
        
        input_name = self._model.get_inputs()[0].name
        token_ids_batch = [self._tokenize(text) for text in data]
        token_inputs = np.array(token_ids_batch, dtype=np.int64)
        outputs = self._model.run({input_name: token_inputs})
        embeddings = outputs[0]
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    def close_session(self):
        self._model.close()

    def init(self):
        self._model.load()
    
    def is_initialized(self):
        return self._model.is_load()
    
    def _tokenize(self, text):
        token_ids = self.tokenizer.encode(text).ids
        return token_ids[:self._max_len] + [0] * (self._max_len - len(token_ids))