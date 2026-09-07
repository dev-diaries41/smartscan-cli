import shutil
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from smartscan.errors import SmartScanError, ErrorCode
from smartscan.providers import (
    TextEmbeddingProvider,
    ImageEmbeddingProvider,
    MiniLmTextEmbedder,
    ClipTextEmbedder,
    ClipImageEmbedder,
    DinoSmallV2ImageEmbedder,
    DistillRobertATextEmbedder
)
from smartscan.models.types import LocalTextEmbeddingModel, LocalImageEmbeddingModel, ModelName
from smartscan.constants import BASE_DIR
from smartscan.models.registry import MODEL_REGISTRY

DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "models")

class ModelManager:
    def __init__(self, root_dir: str = DEFAULT_MODEL_DIR):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def download_model(self, name: ModelName, timeout: int = 30) -> Path:
        model_info = MODEL_REGISTRY[name]
        url = model_info["url"]
        target = self.get_model_path(name)

        if not str(target).startswith(str(self.root_dir)):
            raise SmartScanError(
                "Target path is outside root_dir",
                code=ErrorCode.INVALID_MODEL_PATH,
            )

        target.parent.mkdir(parents=True, exist_ok=True)

        is_dir_model = target.suffix == ""

        if is_dir_model and not url.lower().endswith(".zip"):
            raise SmartScanError(
                "Directory model_path requires a .zip download url",
                code=ErrorCode.INVALID_MODEL_PATH,
            )

        if not is_dir_model and url.lower().endswith(".zip"):
            raise SmartScanError(
                "File model_path cannot use a .zip download url",
                code=ErrorCode.INVALID_MODEL_PATH,
            )

        with tempfile.NamedTemporaryFile(delete=False, dir=str(self.root_dir)) as tmp:
            tmp_path = Path(tmp.name)

            with urllib.request.urlopen(url, timeout=timeout) as resp:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)

        if not is_dir_model:
            tmp_path.replace(target)
            return target

        target.mkdir(parents=True, exist_ok=True)

        extract_tmp = self.root_dir / f".extract_{name}"
        if extract_tmp.exists():
            shutil.rmtree(extract_tmp)

        extract_tmp.mkdir()

        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(extract_tmp)

        tmp_path.unlink(missing_ok=True)

        contents = list(extract_tmp.iterdir())

        if len(contents) == 1 and contents[0].is_dir():
            src_dir = contents[0]
        else:
            src_dir = extract_tmp

        for item in src_dir.iterdir():
            shutil.move(str(item), target / item.name)

        shutil.rmtree(extract_tmp)

        expected_resources = model_info.get("resource_files")
        if expected_resources:
            for f in expected_resources.values():
                if not (target / f).exists():
                    raise SmartScanError(
                        f"Missing expected resource file: {f}",
                        code=ErrorCode.INVALID_MODEL_PATH,
                    )

        return target

    def delete_model(self, name: ModelName) -> None:
        path = self.get_model_path(name)

        if not str(path).startswith(str(self.root_dir)):
            raise SmartScanError(
                "Cannot delete outside root_dir",
                code=ErrorCode.INVALID_MODEL_PATH,
            )

        if not path.exists():
            return

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    def model_exists(self, name: ModelName) -> bool:
        model_info = MODEL_REGISTRY[name]
        path = self.get_model_path(name)

        if path.is_file():
            return path.exists()

        if path.is_dir():
            if not path.exists():
                return False

            expected_resources = model_info.get("resource_files")
            if expected_resources:
                return all((path / f).exists() for f in expected_resources.values())

        return False

    def get_model_path(self, name: ModelName) -> Path:
        model_info = MODEL_REGISTRY[name]
        p = Path(model_info["model_path"])

        return (self.root_dir / p).resolve() if not p.is_absolute() else p.resolve()

    def get_model_download_url(self, name: ModelName) -> str:
        return MODEL_REGISTRY[name]["url"]

    def get_text_embedder(self, model: LocalTextEmbeddingModel) -> TextEmbeddingProvider:
        if model == "clip_vit_b_32_text":
            if not self.model_exists(model):
                print(f"{model} doesn't exsiting. Downloading model now...")
                path = self.download_model(model)
            else:
                path = self.get_model_path(model)
            
            model_info = MODEL_REGISTRY[model]
            resources = model_info['resource_files']
            assert isinstance(resources, dict)

            model_path = path / resources["model"]
            vocab_path = path / resources["vocab"]
            merges_path = path / resources["merges"]          
            return ClipTextEmbedder(model_path, str(vocab_path), str(merges_path))
        
        elif model == "all_minilm_l6_v2":
            if not self.model_exists(model):
                print(f"{model} doesn't exsiting. Downloading model now...")
                path = self.download_model(model)
            else:
                path = self.get_model_path(model)

            model_info = MODEL_REGISTRY[model]
            resources = model_info['resource_files']
            assert isinstance(resources, dict)

            model_path = path / resources["model"]
            vocab_path = path / resources["vocab"]
            return MiniLmTextEmbedder(model_path, str(vocab_path))
        
        elif model == "all_distilroberta_v1":
            if not self.model_exists(model):
                print(f"{model} doesn't exsiting. Downloading model now...")
                path = self.download_model(model)
            else:
                path = self.get_model_path(model)

            model_info = MODEL_REGISTRY[model]
            resources = model_info['resource_files']
            assert isinstance(resources, dict)
            
            model_path = path / resources["model"]
            vocab_path = path / resources["vocab"]
            merges_path = path / resources["merges"]
            return DistillRobertATextEmbedder(model_path, str(vocab_path), str(merges_path))
        
        else:
            raise SmartScanError("Model not supported", code=ErrorCode.UNSUPPORTED_MODEL)

    def get_image_embedder(self, model: LocalImageEmbeddingModel) -> ImageEmbeddingProvider:
        if model == "clip_vit_b_32_image":
            if not self.model_exists(model):
                print(f"{model} doesn't exsiting. Downloading model now...")
                path = self.download_model(model)
                return ClipImageEmbedder(path)

            return ClipImageEmbedder(self.get_model_path(model))

        elif model == "dinov2_small":
            if not self.model_exists(model):
                print(f"{model} doesn't exsiting. Downloading model now...")
                path = self.download_model(model)
                return DinoSmallV2ImageEmbedder(path)

            return DinoSmallV2ImageEmbedder(self.get_model_path(model))

        else:
            raise SmartScanError("Model not supported", code=ErrorCode.UNSUPPORTED_MODEL)