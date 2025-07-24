from pathlib import Path
import os
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import argparse # Import argparse for the main execution block

# Assuming utils.utils contains bgr2ycbcr, to_tensor, set_all_seeds, load_config, create_base_path, find_image_files
from utils.utils import *

# --- PyTorch Dataset Class for On-the-Fly Processing with Sliding Window ---
class ImageSuperResolutionDataset(Dataset):
    """
    PyTorch Dataset for loading HR images and generating LR/HR pairs on-the-fly
    using a sliding window approach for training, and full images for validation.
    Grayscale images are explicitly skipped.
    """
    def __init__(self, hr_image_files: list, upscale_factor: int, interpolation: str, crop_size: int = 96, stride: int = 48, model_name: str = 'SRCNN', mode: str = 'train', lr_image_files: str=None):
        """
        Args:
            hr_image_files (list): List of file paths to the original HR images.
            upscale_factor (int): The super-resolution upscale factor (e.g., 2, 3, 4).
            crop_size (int): The size of HR patches (or target dimensions for validation).
            stride (int): The step size for the sliding window in training mode.
            interpolation (str): Interpolation method for downscaling ('area', 'linear', 'cubic', 'nearest').
            model_name (str): The model type ('SRCNN' or 'SRGAN') to determine preprocessing.
            mode (str): 'train' for sliding window cropping, 'val' for processing full images.
        """
        self.lr_image_files = lr_image_files
        self.hr_image_files = hr_image_files
        self.upscale_factor = upscale_factor
        self.crop_size = crop_size
        self.stride = stride
        self.interpolation_method = self._get_interpolation_flag(interpolation)
        self.model_name = model_name
        self.mode = mode

        self.patch_infos = [] # Stores (hr_image_path, y_start, x_start) for each patch or image
        self.lr_patch_infos = []

        if self.mode == 'train':
            if lr_image_files!=None:
                self.lr_patch_infos=lr_image_files
                self.patch_infos=hr_image_files
                print(f"Total training pair images: {len(self.lr_patch_infos)}")
            else:
                print("Pre-calculating patch coordinates using sliding window for training dataset...")
                for hr_image_path in hr_image_files:
                    try:
                        hr_image = cv2.imread(str(hr_image_path), cv2.IMREAD_UNCHANGED)
                        if hr_image is None:
                            print(f"Warning: Could not read HR image for patch calculation: {hr_image_path}. Skipping.")
                            continue

                        h_orig, w_orig = hr_image.shape[:2]

                        # Generate all possible sliding window patch coordinates
                        for y in range(0, h_orig - self.crop_size + 1, self.stride):
                            for x in range(0, w_orig - self.crop_size + 1, self.stride):
                                self.patch_infos.append((hr_image_path, y, x))
                    except Exception as e:
                        print(f"Error processing image {hr_image_path} for patch calculation: {e}. Skipping.")
                        continue
                print(f"Total {len(self.patch_infos)} patches identified for training.")
        elif self.mode == 'val':
            # For validation, each 'item' corresponds to a full image.
            # Store dummy coordinates (0,0) as they won't be used for cropping in validation mode.
            # Perform initial filtering for grayscale images even for validation paths here.
            for p in hr_image_files:
                try:
                    hr_image = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                    if hr_image is None:
                        print(f"Warning: Could not read HR image for validation: {p}. Skipping.")
                        continue
                    self.patch_infos.append((p, 0, 0)) # Add if it's a valid color image
                except Exception as e:
                    print(f"Error checking image {p} for validation: {e}. Skipping.")
                    continue
            print(f"Total {len(self.patch_infos)} images identified for validation.")
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Must be 'train' or 'val'.")

    def __len__(self) -> int:
        """Returns the total number of patches (for train) or images (for val)."""
        return len(self.patch_infos)

    def _get_interpolation_flag(self, interpolation_str: str) -> int:
        """Converts string interpolation method to OpenCV flag."""
        if interpolation_str == "area":
            return cv2.INTER_AREA
        elif interpolation_str == "linear":
            return cv2.INTER_LINEAR
        elif interpolation_str == "cubic":
            return cv2.INTER_CUBIC
        elif interpolation_str == "nearest":
            return cv2.INTER_NEAREST
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation_str}")

    def __getitem__(self, idx: int):
        """
        Retrieves an HR image/patch based on the mode, generates its corresponding LR,
        and applies model-specific preprocessing.
        """
        
        lr_image_processed = None
        hr_image_processed = None

        if self.mode == 'train':
            if len(self.lr_patch_infos)>0:
                lr_image_path= self.lr_patch_infos[idx]
                hr_image_path= self.hr_image_files[idx]
                lr_image_processed = cv2.imread(str(lr_image_path), cv2.IMREAD_UNCHANGED)
                hr_image_processed = cv2.imread(str(hr_image_path), cv2.IMREAD_UNCHANGED)
            else:
                hr_image_path, y_coord, x_coord = self.patch_infos[idx]
                hr_image_bgr = cv2.imread(str(hr_image_path), cv2.IMREAD_UNCHANGED)
        
                # Extract the specific HR patch using pre-calculated coordinates
                hr_patch = hr_image_bgr[y_coord : y_coord + self.crop_size, x_coord : x_coord + self.crop_size]
                hr_image_processed = np.ascontiguousarray(hr_patch) # Ensure contiguous memory

                # Generate LR patch by downscaling the HR patch
                lr_patch_w = self.crop_size // self.upscale_factor
                lr_patch_h = self.crop_size // self.upscale_factor
                lr_image_processed = cv2.resize(hr_image_processed, (lr_patch_w, lr_patch_h), interpolation=self.interpolation_method)
        
        elif self.mode == 'val':
            # For validation, process the full image (cropped to be divisible by upscale_factor)
            hr_image_path, y_coord, x_coord = self.patch_infos[idx]
            hr_image_bgr = cv2.imread(str(hr_image_path), cv2.IMREAD_UNCHANGED)
            hr_image_processed = np.ascontiguousarray(hr_image_bgr) # HR remains the full original image

            # Generate LR image from the full HR image
            lr_display_w = hr_image_processed.shape[1] // self.upscale_factor
            lr_display_h = hr_image_processed.shape[0] // self.upscale_factor
            
            # LR image downscaled by cubic only
            lr_image_processed = cv2.resize(hr_image_processed, (lr_display_w, lr_display_h), interpolation=self.interpolation_method)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Must be 'train' or 'val'.")

        # --- Model-specific preprocessing (YCbCr for SRCNN, RGB for SRGAN) ---
        lr_tensor = None
        hr_tensor = None

        if self.model_name == "SRCNN":
            lr_y_image = cv2.resize(lr_image_processed, (hr_image_processed.shape[1], hr_image_processed.shape[0]), interpolation=cv2.INTER_CUBIC)

            # Convert to Y channel and normalize [0, 1] for SRCNN
            lr_y_image = bgr2ycbcr(lr_y_image, only_use_y_channel=True)
            hr_y_image = bgr2ycbcr(hr_image_processed, only_use_y_channel=True)

            lr_y_image = lr_y_image.astype(np.float32) / 255.
            hr_y_image = hr_y_image.astype(np.float32) / 255.

            lr_tensor = to_tensor(lr_y_image, range_norm=False, half=False)
            hr_tensor = to_tensor(hr_y_image, range_norm=False, half=False)

            # print(self.model_name)
            # print(f"DEBUG: lr_tensor shape from __getitem__: {lr_tensor.shape}")
            # print(f"DEBUG: hr_tensor shape from __getitem__: {hr_tensor.shape}")
        elif self.model_name == "SRGAN":
            # Convert BGR to RGB and normalize [0, 1] for SRGAN
            lr_image_rgb = cv2.cvtColor(lr_image_processed, cv2.COLOR_BGR2RGB)
            hr_image_rgb = cv2.cvtColor(hr_image_processed, cv2.COLOR_BGR2RGB)

            lr_image_rgb = lr_image_rgb.astype(np.float32) / 255.0
            hr_image_rgb = hr_image_rgb.astype(np.float32) / 255.0

            lr_tensor = to_tensor(lr_image_rgb, range_norm=False, half=False)
            hr_tensor = to_tensor(hr_image_rgb, range_norm=False, half=False)

            lr_tensor = lr_tensor.permute(2, 0, 1)
            hr_tensor = hr_tensor.permute(2, 0, 1)

            # print(f"DEBUG: lr_tensor shape from __getitem__: {lr_tensor.shape}")
            # print(f"DEBUG: hr_tensor shape from __getitem__: {hr_tensor.shape}")
        else:
            raise ValueError(f"Model name '{self.model_name}' not supported for dataset processing.")
        
        return lr_tensor, hr_tensor

# -----------------------------------------------------------------------------

# --- Data Loading/Preparation for Training (using the on-the-fly Dataset) ---
def prepare_data():
    """
    Loads configuration, sets seeds, finds raw HR image files,
    and creates on-the-fly datasets and DataLoaders.
    """
    load_config_data = load_config("../config.yaml") # Ensure this path is correct relative to where prepare_data.py is run
    print("Loaded configuration:", load_config_data)
    set_all_seeds(load_config_data['SEED'])

    base_dirs = create_base_path()

    # Get dataset parameters directly from config for on-the-fly processing
    dataset_params = load_config_data.get("dataset", {
        "upscale_factor": 4,
        "crop_size": 96,
        "stride": 48, # Example stride value (can be same as crop_size or smaller)
        "interpolation": "cubic",
        "train_hr_dir": "data/raw/DIV2K_train_HR", # Path to raw HR training images
        "valid_hr_dir": "data/raw/Set5_val_HR"     # Path to raw HR validation images
    })

    # Find paths to the raw HR image files
    train_hr_image_files = find_image_files(base_dirs["root"] / dataset_params['train_hr_dir'])
    if dataset_params['train_lr_dir']!="":
        train_lr_image_files = find_image_files(base_dirs["root"] / dataset_params['train_lr_dir'])
    else:
        train_lr_image_files=None
    valid_hr_image_files = find_image_files(base_dirs["root"] / dataset_params['valid_hr_dir'])

    # Get the upscale factor and model type from the configuration
    scale_factor = dataset_params['upscale_factor']
    model_type = load_config_data['train'].get('model', 'SRCNN')

    # Create PyTorch Datasets with on-the-fly processing parameters
    train_dataset = ImageSuperResolutionDataset(
        hr_image_files=train_hr_image_files,
        lr_image_files=train_lr_image_files,
        upscale_factor=scale_factor,
        crop_size=dataset_params['crop_size'],
        stride=dataset_params['stride'], # Pass stride here
        interpolation=dataset_params['interpolation'],
        model_name=model_type,
        mode='train'
    )
    valid_dataset = ImageSuperResolutionDataset(
        hr_image_files=valid_hr_image_files,
        upscale_factor=scale_factor,
        interpolation=dataset_params['interpolation'],
        model_name=model_type,
        mode='val'
    )

    # Create PyTorch DataLoaders
    batch_size_train = load_config_data["train"].get(model_type, {}).get('batch_size', 16) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0) # Batch size 1 often for validation metrics

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "scale_factor": scale_factor,
    }

# -----------------------------------------------------------------------------

# --- Main Execution Block (for direct script execution) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility for loading SR datasets with on-the-fly processing.")
    parser.add_argument("--command", type=str, default="load", help="Command to execute: 'load' (default)")

    args = parser.parse_args()

    if args.command == "load":
        print("Starting dataset loading process for training with on-the-fly sliding window generation...")
        results = prepare_data()
        print("\nDataset loading complete. DataLoaders are ready.")
    else:
        print(f"Unknown command: '{args.command}'. Only 'load' is supported in this version.")
        parser.print_help()