from sdk.models.providers.embeddings.embedding_provider import EmbeddingProvider
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
        image_input = self._preprocess(Image.open(data))
        outputs = self._model.run({input_name: image_input})
        embedding = outputs[0][0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    

    def embed_batch(self, data: list[str]):
        """Create vector embeddings for text or image files using an ONNX model."""

        assert self._model.is_load(), "Model not loaded"
        
        input_name = self._model.get_inputs()[0].name
        images = [self._preprocess(Image.open(file)) for file in data]
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
    
    def _preprocess(image: Image.Image):
        SIZE = 224
        MODE = 'RGB'
        MEAN = (0.48145466, 0.4578275, 0.40821073)
        STD = (0.26862954, 0.26130258, 0.27577711)
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
        
        # 5. Transpose to channel-first format (C, H, W)
        img_array = img_array.transpose(2, 0, 1)
        
        # 6. Normalize using the specified mean and std
        mean = np.array(MEAN).reshape(3, 1, 1)
        std = np.array(STD).reshape(3, 1, 1)
        img_array = (img_array - mean) / std
        return img_array.astype(dtype=np.float32)

