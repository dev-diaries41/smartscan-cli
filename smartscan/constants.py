class SupportedExtensions:
    IMAGE = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    TEXT = ('.txt', '.md', '.rst', '.html', '.json')
    VIDEO = ('.mp4', '.mkv', '.webm')

MODEL_REGISTRY = {
    "clip-vit-b-32-image": "image_encoder",
    "clip-vit-b-32-text": "text_encoder",
    "dinov2-small": "image_encoder",
    "inception-resnet-v1": "face_encoder",
    "all-minilm-l6-v2": "text_encoder",
}