from typing import get_args
from PIL import Image
from smartscan.models.model_manager import ModelManager
from smartscan.types import LocalTextEmbeddingModel, LocalImageEmbeddingModel


texts = ["text_1", "text_2"]
model_manager = ModelManager()

def pytest_addoption(parser):
    parser.addoption("--images", args="+nargs")

def test_image_embedders(request):
    images = request.config.getoption("images")
    print(f"Images: {images}")
    model_names = get_args(LocalImageEmbeddingModel)
    for name in model_names:
        model = model_manager.get_image_embedder(name)
        model.init()
        pil_images = [Image.open(img) for img in images]
        embed = model.embed(pil_images)
        print(embed.shape)
        assert embed.shape[0] == len(texts)
        assert embed.shape[1] == model.embedding_dim
        

def test_text_embedders():
    model_names = get_args(LocalTextEmbeddingModel)
    for name in model_names:
        model = model_manager.get_text_embedder(name)
        model.init()
        embed = model.embed(texts)
        print(embed.shape)
        assert embed.shape[0] == len(texts)
        assert embed.shape[1] == model.embedding_dim
        
    