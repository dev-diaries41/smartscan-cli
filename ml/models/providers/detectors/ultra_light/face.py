from ml.models.providers.detectors.detector_provider import DetectorProvider
from PIL import Image
from ml.models.onnx_model import OnnxModel
import numpy as np
import io


class UltraLightFaceDetector(DetectorProvider):
    def __init__(self, model_path: str):
        self._model = OnnxModel(model_path)


    def detect(self, data: str | bytes):
        """Detect faces in a image."""

        assert self._model.is_load(), "Model not loaded"
        
        image = io.BytesIO(data) if isinstance(data, bytes) else data
        input_name = self._model.get_inputs()[0].name
        image_input = self._preprocess(Image.open(image))
        outputs = self._model.run({input_name: image_input})
        scores = outputs[0][0]
        boxes = outputs[1][0]
        return scores, boxes
    
    
    def close_session(self):
        self._model.close()

    def init(self):
        self._model.load()
    
    def is_initialized(self):
        return self._model.is_load()
    
    @staticmethod
    def _preprocess(image: Image.Image):
        SIZE_X = 320
        SIZE_Y = 240
        MODE = 'RGB'
        MEAN = (127, 127, 127)

        # 1. Convert to RGB if not already
        image = image.convert(MODE)
        image = image.resize((SIZE_X, SIZE_Y))
        image = np.array(image)
        image_mean = np.array(MEAN, dtype=np.float32)
        image = (image.astype(np.float32) - image_mean) / 128
        image = image.transpose(2, 0, 1)[None, ...]
        return image.astype(np.float32)

