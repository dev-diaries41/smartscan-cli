from typing import Literal

class SupportedFileTypes:
    IMAGE = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    TEXT = ('.txt', '.md', '.rst', '.html', '.json')
    VIDEO = ('.mp4', '.mkv', '.webm')


ModelName = Literal[
    "clip-vit-b-32-image",
    "clip-vit-b-32-text",
    "dinov2-small",
    "inception-resnet-v1",
    "all-minilm-l6-v2",
]

EncoderType = Literal["image_encoder", "text_encoder", "face_encoder"]

MODEL_REGISTRY: dict[ModelName, EncoderType] = {
    "clip-vit-b-32-image": "image_encoder",
    "clip-vit-b-32-text": "text_encoder",
    "dinov2-small": "image_encoder",
    "inception-resnet-v1": "face_encoder",
    "all-minilm-l6-v2": "text_encoder",
}
