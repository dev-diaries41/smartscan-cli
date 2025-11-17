import json

from dataclasses import dataclass, field, asdict


@dataclass
class SmartScanConfig:
    similarity_threshold: float = 0.7
    target_dirs: list[str] = field(default_factory=list)
    image_encoder_model: str = "dino"
    text_encoder_model: str = "minilm"

def load_config(path: str) -> SmartScanConfig:
    with open(path, "r") as f:
        config = json.load(f)
    
    default = SmartScanConfig()
    for (key, value) in asdict(default).items():
        config.setdefault(key, value)
    return SmartScanConfig(**config)
