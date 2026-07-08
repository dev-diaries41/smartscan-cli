import numpy as np
from smartscan.providers import TextEmbeddingProvider
from smartscan.models.onnx_model import OnnxModel
from smartscan.providers.embeddings.tokenizers import load_clip_tokenizer
from smartscan.errors import SmartScanError, ErrorCode

class ClipTextEmbedder(TextEmbeddingProvider):
    embedding_dim = 512
    max_tokens = 77

    def __init__(self, model_path: str, vocab_path: str, merges_path: str):
        self._model = OnnxModel(model_path)
        self.tokenizer = load_clip_tokenizer(vocab_path, merges_path)


    def embed(self, data: str | list[str]) -> np.ndarray:
        """
        Generate normalized embeddings for one or more input strings.

        Args:
            data: A single string or a list of strings.

        Returns:
            A NumPy array of shape (N, D), where N is the number of input strings
            and D is the embedding dimension. A single input string returns an
            array with shape (1, D).
        """
        if not self.is_initialized():
            raise SmartScanError(
                "Model not loaded",
                code=ErrorCode.MODEL_NOT_LOADED,
                details="Call init method first",
            )

        input_name = self._model.get_inputs()[0].name

        texts = [data] if isinstance(data, str) else data

        token_ids = [self._tokenize(text) for text in texts]
        token_inputs = np.array(token_ids, dtype=np.int64)

        outputs = self._model.run({input_name: token_inputs})
        embeddings = outputs[0]
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    def close_session(self):
        self._model.close()

    def init(self):
        self._model.load()
    
    def is_initialized(self):
        return self._model.is_load()
    
    def _tokenize(self, text: str):
        token_ids = self.tokenizer.encode(text).ids
        return token_ids[:self.max_tokens] + [0] * (self.max_tokens - len(token_ids))