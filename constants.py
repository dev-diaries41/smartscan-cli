import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_ENCODER_PATH = os.path.join(BASE_DIR, 'models/clip_text_encoder_quant.onnx')
IMAGE_ENCODER_PATH = os.path.join(BASE_DIR, 'models/clip_image_encoder_quant.onnx')
MINILM_MODEL_PATH = os.path.join(BASE_DIR, 'models/minilm_sentence_transformer_quant.onnx')


