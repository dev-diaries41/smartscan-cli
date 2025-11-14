from smartscan.ml.models.base_model import BaseModel
import onnxruntime as ort
import numpy as np

class OnnxModel(BaseModel):
    def __init__(self, model_path: str):
        self.ort_session = None
        self.model_path = model_path

    def load(self):
        self.ort_session =  ort.InferenceSession(self.model_path)


    def is_load(self) -> bool:
        return self.ort_session is not None
    
    def close(self):
        self.ort_session = None

    def get_inputs(self):
        return self.ort_session.get_inputs()

    def run(self, inputs: dict) -> list[np.ndarray]:
        return self.ort_session.run(None, inputs)