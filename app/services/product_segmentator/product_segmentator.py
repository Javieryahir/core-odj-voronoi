import shutil
import os
import cv2
import numpy as np
import uuid
from .runner import run_parallel


BASE_OUTPUT_PATH = os.path.join(os.getcwd(), "tmp", "output")
os.makedirs(BASE_OUTPUT_PATH, exist_ok=True) # Make sure the temporary output folder exists


def bytes_to_cv2(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np


import os
import base64

# This function also converts images found in subfolders
def folder_to_base64(folder_path: str) -> list[str]:
    base64_images = []
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    
    if not os.path.exists(folder_path):
        raise ValueError(f"The specified folder does not exist: {folder_path}")

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(valid_extensions):
                file_path = os.path.join(root, filename)
                
                try:
                    with open(file_path, "rb") as image_file:
                        binary_data = image_file.read()
                        base64_string = base64.b64encode(binary_data).decode('utf-8')
                        base64_images.append(base64_string)
                except Exception as e:
                    print(f"Could not process {file_path}: {e}")

    return base64_images


def product_segmentation(
        image: bytes, 
        remove_output_files=True, 
        no_box_filtering=False, 
        area_percentile=99.0, 
        iou_merge_threshold=0.3, 
        aspect_ratio_sigma=2.0, 
        duplicate_threshold=0.7, 
        diagonal_gap_ratio=0.3, 
        min_group_size=2,
        max_box_ratio=0.9
        ) -> None:
    img_np = bytes_to_cv2(image)
    if img_np is None:
        raise ValueError("Image not found.")

    request_uuid = str(uuid.uuid4())
    output_path = os.path.join(BASE_OUTPUT_PATH, request_uuid)
    run_parallel(
        img_np,
        output_root=output_path,
        disable_box_filtering=no_box_filtering,
        area_percentile=area_percentile,
        iou_merge_threshold=iou_merge_threshold,
        aspect_ratio_sigma=aspect_ratio_sigma,
        duplicate_threshold=duplicate_threshold,
        diagonal_gap_ratio=diagonal_gap_ratio,
        min_group_size=min_group_size,
        max_box_ratio=max_box_ratio,
    )

    images = folder_to_base64(os.path.join(output_path, "result-images"))

    if remove_output_files and os.path.exists(output_path):
        shutil.rmtree(output_path)

    return images
    