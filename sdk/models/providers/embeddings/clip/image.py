from sdk.models.providers.embeddings.embedding_provider import EmbeddingProvider
from utils import preprocess
from PIL import Image
from sdk.models.onnx_model import OnnxModel
import numpy as np

class ClipImageEmbedder(EmbeddingProvider):
    def __init__(self, model_path: str):
        self._model = OnnxModel(model_path)
        self._embedding_dim = 512

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, data: str):
        """Create vector embeddings for text or image files using an ONNX model."""

        assert self._model.is_load(), "Model not loaded"
        
        input_name = self._model.get_inputs()[0].name
        image_input = preprocess(Image.open(data))
        outputs = self._model.run({input_name: image_input})
        embedding = outputs[0][0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    

    def embed_batch(self, data: list[str]):
        """Create vector embeddings for text or image files using an ONNX model."""

        assert self._model.is_load(), "Model not loaded"
        
        input_name = self._model.get_inputs()[0].name
        images = [preprocess(Image.open(file)) for file in data]
        image_inputs = np.stack(images, axis=0)
        outputs = self._model.run({input_name: image_inputs})
        embeddings = outputs[0]
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    def close_session(self):
        self._model.close()

    def init(self):
        self._model.load()
    
    def is_initialized(self):
        return self._model.is_load()