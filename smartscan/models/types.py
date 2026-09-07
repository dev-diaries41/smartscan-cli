from typing import Literal, TypeAlias, TypedDict, Optional, NotRequired

LocalTextEmbeddingModel: TypeAlias = Literal["all_minilm_l6_v2", "clip_vit_b_32_text", "all_distilroberta_v1"]
LocalImageEmbeddingModel: TypeAlias = Literal["clip_vit_b_32_image", "dinov2_small"]
LocalFaceEmbeddingModel: TypeAlias = Literal["inception_resnet_v1"]

ModelName = Literal[LocalTextEmbeddingModel,LocalImageEmbeddingModel, LocalFaceEmbeddingModel]

class ResourceFiles(TypedDict):
    model: str
    merges: NotRequired[str]
    vocab: NotRequired[str]

class ModelInfo(TypedDict):
    url: str
    model_path: str             
    resource_files: Optional[ResourceFiles]
    file_hash: Optional[str]
