import os
import pickle
import shutil
import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def read_text_file(filepath: str):
   assert os.path.isfile(filepath), "Invalid file"
   with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()       
  

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
    

def save_embedding(filepath: str, embedding: np.ndarray):
    """Saves a tensor embedding to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump(embedding, f)

def load_embedding(filepath: str) -> np.ndarray:
    """Loads a tensor embedding from a file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def has_one_week_passed(file_path: str) -> bool:
    last_modified_timestamp = os.path.getmtime(file_path)    
    last_modified_date = datetime.datetime.fromtimestamp(last_modified_timestamp)    
    current_date = datetime.datetime.now()    
    days_since_modified = (current_date - last_modified_date).days
    return days_since_modified % 7 == 0


def generate_prototype_embedding(embeddings):    
    embeddings_tensor = np.stack(embeddings, dim=0)
    prototype = np.mean(embeddings_tensor, dim=0)
    prototype /= np.linalg.norm(prototype)
    return prototype


def get_files_from_dir(dir_path, limit = 30):
    assert os.path.isdir(dir_path), "Invalid directory"
    return [ os.path.join(dir_path, filename) for filename in os.listdir(dir_path)[:limit]]    


def nms_numpy(boxes, scores, iou_threshold):
    """Simple NMS in NumPy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:][iou <= iou_threshold]

    return keep

def draw_boxes(
    image: Image.Image,
    boxes: np.ndarray,
    scores: np.ndarray,
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.3,
    outline: str = "green",
    width: int = 2,
    font: ImageFont.ImageFont = None,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    if font is None:
        try:
            font = ImageFont.truetype("arial.ttf", size=14)
        except IOError:
            font = ImageFont.load_default()

    img_w, img_h = image.size

    # Filter by confidence
    keep_indices = np.where(scores >= conf_threshold)[0]
    if keep_indices.size == 0:
        return image

    boxes_px = (boxes[keep_indices] * [img_w, img_h, img_w, img_h]).astype(int)
    scores_keep = scores[keep_indices]

    # NMS
    keep_nms = nms_numpy(boxes_px, scores_keep, nms_threshold)
    filtered_boxes = boxes_px[keep_nms]
    filtered_scores = scores_keep[keep_nms]

    # Draw
    for (x1, y1, x2, y2), score in zip(filtered_boxes, filtered_scores):
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)
        text = f"{score:.2f}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])        
        draw.rectangle([x1, y1 - text_size[1], x1 + text_size[0], y1], fill=outline)
        draw.text((x1, y1 - text_size[1]), text, fill="white", font=font)

    return image


def crop_faces(image: Image.Image, boxes: np.ndarray, scores: np.ndarray, conf_threshold: float = 0.5, nms_threshold: float = 0.3):
    img_w, img_h = image.size
    keep_indices = np.where(scores >= conf_threshold)[0]
    if keep_indices.size == 0:
        return []

    boxes_px = (boxes[keep_indices] * [img_w, img_h, img_w, img_h]).astype(int)
    scores_keep = scores[keep_indices]

    keep_nms = nms_numpy(boxes_px, scores_keep, nms_threshold)
    filtered_boxes = boxes_px[keep_nms]

    cropped_faces = [image.crop((x1, y1, x2, y2)) for (x1, y1, x2, y2) in filtered_boxes]
    return cropped_faces
