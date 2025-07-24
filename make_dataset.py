
from pathlib import Path
import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import argparse
import time
from datetime import datetime
import json

def process_image(image_path, crop_size, stride, upscale_factor, interpolation, lr_dir, hr_dir=None):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    h, w = image.shape[:2]
    img_name = image_path.stem
    patch_id = 1

    if hr_dir is not None:
        for y in range(0, h - crop_size + 1, stride):
            for x in range(0, w - crop_size + 1, stride):
                hr_patch = image[y:y + crop_size, x:x + crop_size]
                hr_patch = np.ascontiguousarray(hr_patch)

                # Simulate LR patch
                # cv2.INTER_AREA: Generally preferred for shrinking images.
                # cv2.INTER_LINEAR: Good for zooming; default method.
                # cv2.INTER_CUBIC: Slower but produces higher quality results for zooming.
                # cv2.INTER_NEAREST: Simple and fast, but can produce blocky results.
                if interpolation == "area":
                    lr_patch = cv2.resize(hr_patch, (crop_size // upscale_factor, crop_size // upscale_factor), interpolation=cv2.INTER_AREA)
                    # lr_patch = cv2.resize(lr_patch, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
                elif interpolation == "linear":
                    lr_patch = cv2.resize(hr_patch, (crop_size // upscale_factor, crop_size // upscale_factor), interpolation=cv2.INTER_LINEAR)
                    # lr_patch = cv2.resize(lr_patch, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
                elif interpolation == "cubic":
                    lr_patch = cv2.resize(hr_patch, (crop_size // upscale_factor, crop_size // upscale_factor), interpolation=cv2.INTER_CUBIC)
                    # lr_patch = cv2.resize(lr_patch, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
                elif interpolation == "nearest":
                    lr_patch = cv2.resize(hr_patch, (crop_size // upscale_factor, crop_size // upscale_factor), interpolation=cv2.INTER_NEAREST)
                    # lr_patch = cv2.resize(lr_patch, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

                filename=f"{img_name}_{patch_id:04d}.png"

                
                lr_save_path = lr_dir / filename
                cv2.imwrite(str(lr_save_path), lr_patch)
                hr_save_path = hr_dir / filename
                cv2.imwrite(str(hr_save_path), hr_patch)
                patch_id += 1

    else:
        if interpolation == "area":
            lr_patch = cv2.resize(image, (w // upscale_factor, h // upscale_factor), interpolation=cv2.INTER_AREA)
        elif interpolation == "linear":
            lr_patch = cv2.resize(image, (w // upscale_factor, h // upscale_factor), interpolation=cv2.INTER_LINEAR)
        elif interpolation == "cubic":
            lr_patch = cv2.resize(image, (w // upscale_factor, h // upscale_factor), interpolation=cv2.INTER_CUBIC)
        elif interpolation == "nearest":
            lr_patch = cv2.resize(image, (w // upscale_factor, h // upscale_factor), interpolation=cv2.INTER_NEAREST)

        filename=f"{img_name}.png"

        lr_save_path = lr_dir / filename
        cv2.imwrite(str(lr_save_path), lr_patch)


def prepare_dataset(args):
    base_dir = Path(args.output_dir)
    start_time = time.time()
    root = Path(args.output_dir) / f"{args.interpolation}_X{args.upscale_factor}_pixels{args.crop_size}_stride{args.stride}"
    if args.train_dir:
        data_dir=Path(args.train_dir)
        lr_dir=root / "train" / "lr"
        hr_dir=root / "train" / "hr"
        os.makedirs(lr_dir, exist_ok=True)
        os.makedirs(hr_dir, exist_ok=True)
    elif args.valid_dir:
        data_dir=Path(args.valid_dir)
        lr_dir=root / "valid" / "lr"
        hr_dir=None
        os.makedirs(lr_dir, exist_ok=True)
    images_files = list(data_dir.glob("*.[jp][pn]g")) + list(data_dir.glob("*.bmp"))
    for image_path in tqdm(images_files, desc="Processing valid images"):
        process_image(image_path, args.crop_size, args.stride, args.upscale_factor, args.interpolation, lr_dir, hr_dir)
    duration = time.time() - start_time
    log_path = os.path.join(root, f"log.json")
    log_configuration_json(log_path, args, duration)    


def log_configuration_json(log_path, args, duration_sec):
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "runtime_seconds": round(duration_sec, 2)
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare paired LR-HR dataset with train/valid split.")
    parser.add_argument("--train_dir", type=str, default=None, help="Optional directory of predefined training HR images.")
    parser.add_argument("--valid_dir", type=str, default=None, help="Optional directory of predefined validation HR images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory.")
    parser.add_argument("--crop_size", type=int, default=96, help="HR patch size.")
    parser.add_argument("--stride", type=int, default=48, help="Stride for sliding crop window.")
    parser.add_argument("--upscale_factor", type=int, choices=[2, 3, 4], default=4, help="Upscaling factor.")
    parser.add_argument("--interpolation", type=str, choices=["area", "linear", "cubic", "nearest"], default="area", help="Interpolation method.")
    args = parser.parse_args()

    prepare_dataset(args)