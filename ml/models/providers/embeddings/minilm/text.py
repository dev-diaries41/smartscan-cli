from ml.models.providers.embeddings.embedding_provider import EmbeddingProvider
from ml.models.onnx_model import OnnxModel
from ml.models.providers.embeddings.minilm.tokenizer import load_minilm_tokenizer
import numpy as np
from pathlib import Path
from utils import read_text_file
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.txt")

class MiniLmTextEmbedder(EmbeddingProvider):
    def __init__(self, model_path: str):
        self._model = OnnxModel(model_path)
        self._embedding_dim = 384
        self._max_len = 128
        self.tokenizer = load_minilm_tokenizer(VOCAB_PATH)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, data: str | Path):
        """Create vector embeddings for text using an ONNX model."""

        assert self._model.is_load(), "Model not loaded"
        
        text = read_text_file(data) if isinstance(data, Path) else data
        input_name = self._model.get_inputs()[0].name
        token_ids = self._tokenize(text)
        attention_mask = [1 if id != 0 else 0 for id in token_ids]
        token_input = np.array([token_ids], dtype=np.int64)
        mask_input = np.array([attention_mask], dtype=np.int64)

        token_input = np.array([token_ids], dtype=np.int64)
        outputs = self._model.run({input_name: token_input, "attention_mask": mask_input})
        embedding = outputs[0][0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    

    def embed_batch(self, data: list[str] | list[Path]):
        """Create vector embeddings for batch of text files using an ONNX model."""

        assert self._model.is_load(), "Model not loaded"
        
        input_names = self._model.get_inputs()
        token_ids_batch = [self._tokenize(read_text_file(item) if isinstance(item, Path) else item) for item in data]
        attention_mask_batch = [[1 if id != 0 else 0 for id in token_ids] for token_ids in token_ids_batch]

        token_inputs = np.array(token_ids_batch, dtype=np.int64)
        mask_inputs = np.array(attention_mask_batch, dtype=np.int64)

        outputs = self._model.run({input_names[0].name: token_inputs, input_names[1].name: mask_inputs})
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