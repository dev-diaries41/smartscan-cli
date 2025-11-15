import os
import pickle
import shutil
import datetime
import numpy as np
from pathlib import Path

def read_text_file(filepath: str):
   assert os.path.isfile(filepath), "Invalid file"
   with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()       
  

def load_dir_list(dirlist_file):
    """Read the list of directory paths from the provided text file."""
    with open(dirlist_file, "r") as f:
        dirs = [line.strip() for line in f if line.strip() and os.path.isdir(line.strip())]
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


def save_embedding(filepath: str, embedding: np.ndarray):
    """Saves a tensor embedding to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump(embedding, f)

def load_embedding(filepath: str) -> np.ndarray:
    """Loads a tensor embedding from a file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    

def get_days_since_last_modified(file_path: str) -> int:
    last_modified_timestamp = os.path.getmtime(file_path)    
    last_modified_date = datetime.datetime.fromtimestamp(last_modified_timestamp)    
    current_date = datetime.datetime.now()    
    days_since_modified = (current_date - last_modified_date).days
    return days_since_modified 


def get_files_from_dirs(dirs: list[str], skip_patterns: list[str] = [], limit: int | None = None) -> list[str]:
    if not isinstance(dirs, list):
        raise ValueError("Invalid list of directories")
    
    paths = []

    def get_files(base: Path):
        nonlocal paths
        try:
            for entry in base.iterdir():
                if entry.is_dir() and any(entry.match(pat) for pat in skip_patterns):
                    continue
                if entry.is_file():
                    if limit is not None and len(paths) >= limit:
                        return
                    paths.append(str(entry.resolve()))
                elif entry.is_dir():
                    get_files(entry)
        except PermissionError:
            print(f"[Skipped] Permission denied: {base}")
            return

    for d in dirs:
        root_dir = Path(d)
        if root_dir.is_dir():
            get_files(root_dir)
    
    return paths
