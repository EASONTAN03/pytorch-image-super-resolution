import numpy as np
import pandas as pd
import torch
import random
import os
import yaml
from pathlib import Path
import pickle
import json
import matplotlib.pyplot as plt
import re
import cv2

# Set random seeds for reproducibility
def set_all_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_base_path():
    # Get absolute project root (assumes this script lives in src/utils/)
    project_root = Path(__file__).resolve().parents[2]

    # Define all directories relative to project root
    base_dirs = {
        "root": project_root,
        "results": project_root / "results",
        "plots": project_root / "results" / "plots",
        "models": project_root / "results" / "models",
        "data": project_root / "data",
        "raw_data": project_root / "data" / "raw",
        "processed_data": project_root / "data" / "processed"
    }

    # Create directories if they don't exist
    for path in base_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return base_dirs

def create_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def find_image_files(folder_path):
    """
    Finds all image files (PNG, JPG, JPEG) within a specified folder.

    Args:
        folder_path (str): The path to the folder to search.

    Returns:
        list: A list of full paths to all image files found in the folder.
              Returns an empty list if the folder does not exist or
              no image files are found.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return []
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return image_files

def retrieve_scale_factor(input_value):
    # Your previously defined function
    if isinstance(input_value, int):
        return input_value
    elif isinstance(input_value, str):
        match = re.search(r'X(\d+)', input_value)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None
    return None

def bgr2ycbcr(image: np.ndarray, only_use_y_channel: bool = True) -> np.ndarray:
    """
    Convert a BGR image to YCbCr color space and return either:
    - the Y (luminance) channel, or
    - the full YCbCr image.

    Args:
        image (np.ndarray): Input image in BGR format (uint8 or float32).
        only_use_y_channel (bool): If True, return only the Y channel.

    Returns:
        np.ndarray: Y channel or full YCbCr image.
    """
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)

    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    if only_use_y_channel:
        return ycbcr[:, :, 0]
    return ycbcr

def to_tensor(image, range_norm=False, half=False):
    """
    Converts an image (NumPy array) into a PyTorch Tensor.
    Handles grayscale images by adding a channel dimension.
    Optionally normalizes the range to [-1, 1] and converts to half-precision.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        range_norm (bool, optional): If True, normalize the tensor to the range [-1, 1]. Defaults to False.
        half (bool, optional): If True, convert the tensor to half-precision (torch.float16). Defaults to False.

    Returns:
        torch.Tensor: The converted image as a PyTorch Tensor.
    """
    tensor = torch.from_numpy(image).float() # Convert to float32 tensor

    if image.ndim == 2: # For grayscale images (Y channel), add a channel dimension (C, H, W)
        tensor = tensor.unsqueeze(0) # Becomes (1, H, W)

    if range_norm: # Normalize to [-1, 1] if required
        # Assumes input image pixel values are already in [0, 1] as per common preprocessing
        tensor = tensor * 2.0 - 1.0

    if half: # Convert to half-precision if required
        tensor = tensor.half()

    return tensor

def save_training_graph(train_data: list, val_data: list, title: str, ylabel: str, save_path: str):
    """
    Saves a plot of training and validation data (e.g., loss or metric) over epochs.

    Args:
        train_data (list): List of training values per epoch.
        val_data (list): List of validation values per epoch.
        title (str): Title of the plot.
        ylabel (str): Label for the Y-axis.
        save_path (str): Full path including filename (e.g., 'plots/train_loss.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_data, label='Training')
    plt.plot(val_data, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Ensure the directory for saving the plot exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True) 
    
    plt.savefig(save_path)
    plt.close() # Close the plot to free memory
    print(f"Graph saved to: {save_path}")

def append_results_to_csv(csv_path: str, data_row: dict):
    """
    Appends a row of results to a CSV file. Creates the file and writes headers if it doesn't exist.

    Args:
        csv_path (str): Full path to the CSV file (e.g., 'logs/model_results.csv').
        data_row (dict): Dictionary where keys are column headers and values are row data.
                         Example: {"timestamp": "...", "model_name": "...", "best_val_psnr": "..."}
    """
    df = pd.DataFrame([data_row]) # Create a DataFrame from the single row of data
    
    # Ensure the directory for the CSV file exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) 

    if not os.path.exists(csv_path):
        # If the file does not exist, write with header
        df.to_csv(csv_path, index=False)
        print(f"Created new CSV and saved results to: {csv_path}")
    else:
        # If the file exists, append without header
        df.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"Appended results to CSV at: {csv_path}")

def save_log(log_path, log_entry):
    # Append to log.json
    if log_path.exists():
        with open(log_path, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(log_path, "w") as f:
        json.dump(logs, f, indent=4)

    print("Logs updated.")