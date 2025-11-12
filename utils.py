import os
import torch
import pickle
import shutil
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