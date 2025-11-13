from ml.models.providers.embeddings.embedding_provider import EmbeddingProvider
from PIL import Image
from ml.models.onnx_model import OnnxModel
import numpy as np
import io


class InceptionResnetFaceEmbedder(EmbeddingProvider):
    def __init__(self, model_path: str):
        self._model = OnnxModel(model_path)

    @property
    def embedding_dim(self) -> int:
        return 512

    def embed(self, data: str | bytes):
        """Create vector embeddings for text or image files using an ONNX model."""

        assert self._model.is_load(), "Model not loaded"
        
        image = io.BytesIO(data) if isinstance(data, bytes) else data
        input_name = self._model.get_inputs()[0].name
        image_input = self._preprocess(Image.open(image))
        outputs = self._model.run({input_name: image_input})
        embedding = outputs[0][0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    

    def embed_batch(self, data: list[bytes] | list[str]):
        """Create vector embeddings for text or image files using an ONNX model."""

        assert self._model.is_load(), "Model not loaded"
        
        input_name = self._model.get_inputs()[0].name
        images = [self._preprocess(Image.open(io.BytesIO(item) if isinstance(item, bytes) else item)) for item in data]
        image_inputs = np.concatenate(images, axis=0)
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
    
    @staticmethod
    def _preprocess(image: Image.Image):
        SIZE = 160
        MODE = 'RGB'
        MEAN = (0.485, 0.456, 0.406)
        STD = (0.229, 0.224, 0.225)
        INTERPOLATION = Image.BICUBIC

        # 1. Convert to RGB if not already
        image = image.convert(MODE)
        
        # 2. Resize based on the shortest edge
        w, h = image.size
        # Compute scaling factor so that the shortest edge becomes SIZE
        scale = SIZE / min(w, h)
        new_w, new_h = round(w * scale), round(h * scale)
        image = image.resize((new_w, new_h), INTERPOLATION)
        
        # 3. Center crop to SIZE x SIZE
        left = (new_w - SIZE) // 2
        top = (new_h - SIZE) // 2
        image = image.crop((left, top, left + SIZE, top + SIZE))
        
        # 4. Convert to NumPy array and scale pixel values to [0, 1]
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # 5. Transpose to channel-first format (B, C, H, W)
        img_array = img_array.transpose(2, 0, 1)[None, ...]
        
        # 6. Normalize using the specified mean and std
        mean = np.array(MEAN).reshape(3, 1, 1)
        std = np.array(STD).reshape(3, 1, 1)
        img_array = (img_array - mean) / std
        return img_array.astype(dtype=np.float32)

