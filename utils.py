import os
import torch
import pickle
import shutil
from PIL import Image
import numpy as np
import datetime

def read_text_file(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file: {e}"

def load_dir_list(dirlist_file):
    """Read the list of directory paths from the provided text file."""
    with open(dirlist_file, "r") as f:
        dirs = [line.strip() for line in f if line.strip()]
    for d in dirs:
        if not os.path.isdir(d):
            print(f"Warning: '{d}' is not a valid directory.")
    return dirs


def move_file(file_path, target_dir):
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return None

    if not os.path.isdir(target_dir):
        print(f"Error: Target directory '{target_dir}' does not exist.")
        return None

    try:
        new_path = os.path.join(target_dir, os.path.basename(file_path))
        shutil.move(file_path, new_path)
        print(f"File moved to {new_path}")
        return new_path
    except Exception as e:
        print(f"Error moving file: {e}")
        return None
    

def save_embedding(filepath: str, embedding: torch.Tensor):
    """Saves a tensor embedding to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump(embedding, f)

def load_embedding(filepath: str) -> torch.Tensor:
    """Loads a tensor embedding from a file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def has_one_week_passed(file_path: str) -> bool:
    last_modified_timestamp = os.path.getmtime(file_path)    
    last_modified_date = datetime.datetime.fromtimestamp(last_modified_timestamp)    
    current_date = datetime.datetime.now()    
    days_since_modified = (current_date - last_modified_date).days
    return days_since_modified % 7 == 0


def preprocess(image: Image.Image) -> torch.Tensor:
    """
    Preprocesses the input PIL image to match the CLIP model input requirements.
    
    Steps:
    1. Convert image to RGB.
    2. Resize image so that the shortest edge is SIZE using bicubic interpolation.
    3. Center crop to SIZE x SIZE.
    4. Convert image to a NumPy array, scale pixel values to [0, 1].
    5. Transpose to channel-first format.
    6. Normalize using the OpenAI dataset mean and std.
    
    Args:
        image (PIL.Image.Image): The input image.
        
    Returns:
        torch.Tensor: The preprocessed image as a float32 tensor with shape (3, SIZE, SIZE).
    """
    SIZE = 224
    MODE = 'RGB'
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)
    INTERPOLATION = Image.BICUBIC  # 'bicubic'
    RESIZE_MODE = 'shortest'
    FILL_COLOR = 0  # not used here since we resize based on shortest edge

    # 1. Convert to RGB if not already
    image = image.convert(MODE)
    
    # 2. Resize based on the shortest edge
    w, h = image.size
    # Compute scaling factor so that the shortest edge becomes SIZE
    scale = SIZE / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    image = image.resize((new_w, new_h), INTERPOLATION)
    
    # 3. Center crop to SIZE x SIZE
    left = (new_w - SIZE) // 2
    top = (new_h - SIZE) // 2
    image = image.crop((left, top, left + SIZE, top + SIZE))
    
    # 4. Convert to NumPy array and scale pixel values to [0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # 5. Transpose to channel-first format (C, H, W)
    img_array = img_array.transpose(2, 0, 1)
    
    # 6. Normalize using the specified mean and std
    mean = np.array(MEAN).reshape(3, 1, 1)
    std = np.array(STD).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    
    img_array = torch.from_numpy(img_array)
    return img_array.to(torch.float32)
