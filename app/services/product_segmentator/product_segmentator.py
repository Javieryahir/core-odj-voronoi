import shutil
import os
import cv2
import numpy as np
import uuid
from .runner import run_parallel


CWD = os.path.join(os.getcwd(), "tmp")
BASE_OUTPUT_PATH = os.path.join(CWD, "output")
os.makedirs(BASE_OUTPUT_PATH, exist_ok=True) # Make sure the temporary output folder exists


def bytes_to_cv2(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np


import os
import base64

def folder_to_base64(folder_path: str) -> list[str]:
    base64_images = []
    
    if not os.path.exists(folder_path):
        raise ValueError("The specified folder does not exist")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".jpg"):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, "rb") as image_file:
                    binary_data = image_file.read()
                    base64_string = base64.b64encode(binary_data).decode('utf-8')
                    base64_images.append(base64_string)
            except Exception as e:
                print(f"Could not process {filename}: {e}")
                raise e

    return base64_images


def product_segmentation(image: bytes, no_box_filtering=False, remove_output_files=True) -> None:
    img_np = bytes_to_cv2(image)
    if img_np is None:
        raise ValueError("Image not found.")

    request_uuid = str(uuid.uuid4())
    output_path = os.path.join(BASE_OUTPUT_PATH, request_uuid)
    run_parallel(
        img_np,
        output_root=output_path,
        disable_box_filtering=no_box_filtering
    )

    images = folder_to_base64(os.path.join(output_path, "result-images"))

    if remove_output_files and os.path.exists(output_path):
        shutil.rmtree(output_path)

    return images
    