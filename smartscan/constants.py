import os 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Models
TEXT_ENCODER_PATH = os.path.join(BASE_DIR, 'models/clip_text_encoder_quant.onnx')
IMAGE_ENCODER_PATH = os.path.join(BASE_DIR, 'models/clip_image_encoder_quant.onnx')
MINILM_MODEL_PATH = os.path.join(BASE_DIR, 'models/minilm_sentence_transformer_quant.onnx')
DINO_V2_SMALL_MODEL_PATH = os.path.join(BASE_DIR, 'models/dinov2_small_quant.onnx')
ULTRA_LIGHT_FACE_DETECTOR_MODEL_PATH = os.path.join(BASE_DIR, 'models/face_detect.onnx')
INCEPTION_RESNET_MODEL_PATH = os.path.join(BASE_DIR, 'models/inception_resnet_v1_quant.onnx')

# Providers
EMBEDDING_PROVIDERS_DIR = os.path.join(BASE_DIR, "smartscan/ml/providers/embeddings")

# DB
DB_DIR = os.path.join(BASE_DIR, "db")
SCAN_HISTORY_DB = os.path.join(BASE_DIR, "db/scan_history.db")


