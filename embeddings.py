import torch
from PIL import Image
import os
from utils import read_text_file, preprocess
import onnxruntime as ort
from onnx_utils.convert import IMAGE_ENCODER_PATH, TEXT_ENCODER_PATH


def get_embedding(model, tokenizer, file_path):
    """Create vector embeddings for text or image files."""
    valid_img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    valid_txt_extensions = ('.txt', '.md', '.rst', '.html', '.json')
    file_embedding = None

    try:
        if file_path.lower().endswith(valid_img_extensions):
            image = Image.open(file_path).convert("RGB")
            image_input: torch.Tensor = preprocess(image).unsqueeze(0).to("cpu")
            with torch.no_grad():
                file_embedding = model.encode_image(image_input)
        elif file_path.lower().endswith(valid_txt_extensions):
            text = read_text_file(file_path)
            tokenized_text = tokenizer(text)
            with torch.no_grad():
                file_embedding = model.encode_text(tokenized_text)
        else:
            raise ValueError(f"Unsupported file type. Valid extensions: {valid_img_extensions}, {valid_txt_extensions}")

      
        file_embedding /= file_embedding.norm(dim=-1, keepdim=True)
        return file_embedding.squeeze(0)

    except Exception as e:
        raise RuntimeError(f"Error processing embedding: {e}")


def get_embedding_onnx(tokenizer, file_path: str):
    """Create vector embeddings for text or image files using an ONNX model."""
    valid_img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    valid_txt_extensions = ('.txt', '.md', '.rst', '.html', '.json')
    embedding = None

    try:
        if file_path.lower().endswith(valid_img_extensions):
            ort_session = ort.InferenceSession(IMAGE_ENCODER_PATH)
            input_name = ort_session.get_inputs()[0].name
            image_input = preprocess(Image.open(file_path)).unsqueeze(0).to("cpu")
            outputs = ort_session.run(None, {input_name: image_input.numpy()})
        elif file_path.lower().endswith(valid_txt_extensions):
            text = read_text_file(file_path)
            token_input = tokenizer(text)
            ort_session = ort.InferenceSession(TEXT_ENCODER_PATH)
            input_name = ort_session.get_inputs()[0].name
            outputs = ort_session.run(None, {input_name: token_input.numpy()})
        else:
            raise ValueError(f"Unsupported file type. Valid extensions: {valid_img_extensions}, {valid_txt_extensions}")

        # Convert output to a PyTorch tensor and normalize
        embedding = torch.tensor(outputs[0][0])
        embedding = embedding / embedding.norm()

        return embedding

    except Exception as e:
        raise RuntimeError(f"Error processing embedding: {e}")


def aggregate_embeddings_from_dir(tokenizer, dir_path, model = None):
    """Compute and average embeddings for all valid image and text files in a directory."""
    
    embeddings = []

    try:
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if model is not None:
                emb = get_embedding(model, tokenizer, file_path)
            else:
                emb = get_embedding_onnx(tokenizer, file_path)
            embeddings.append(emb)
                
    except Exception as e:
        raise RuntimeError(f"Error aggregating embeddings from {dir_path}: {e}")

    if not embeddings:
        raise ValueError(f"No valid files found in the directory {dir_path}.")

    embeddings_tensor = torch.stack(embeddings, dim=0)
    prototype = torch.mean(embeddings_tensor, dim=0)
    prototype /= prototype.norm()
    return prototype

def calculate_cosine_similarity(embedding1, embedding2):
    try:
        with torch.no_grad():
            similarity = embedding1 @ embedding2
    except Exception as e:
        raise RuntimeError(f"Error computing similarity: {e}")
    return similarity.item()