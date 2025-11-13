import os 

MODEL_NAME = 'ViT-B-32'
PRETRAINED = 'laion2b_s34b_b79k'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_ENCODER_PATH = os.path.join(BASE_DIR, 'clip_text_encoder_quant.onnx')
IMAGE_ENCODER_PATH = os.path.join(BASE_DIR, 'clip_image_encoder_quant.onnx')
MINILM_MODEL_PATH = os.path.join(BASE_DIR, 'minilm_sentence_transformer_quant.onnx')


