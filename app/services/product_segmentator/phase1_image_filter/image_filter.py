import cv2
import numpy as np


def apply_gaussian_filter(image: np.ndarray,
                          kernel_size: int = 5,
                          sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian low-pass filter to reduce noise
    while preserving useful gradient structure.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def apply_clahe_lab(image: np.ndarray,
                    clip_limit: float = 2.0,
                    tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE on the L channel in LAB space
    to normalize illumination.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def convert_color_space(image: np.ndarray,
                        space: str = "LAB") -> np.ndarray:
    """
    Convert image to LAB or HSV for later segmentation.
    """
    if space.upper() == "LAB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    elif space.upper() == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError("Unsupported color space. Use 'LAB' or 'HSV'.")


def phase1_preprocess(image_path: str,
                      output_path: str = None,
                      color_space: str = "LAB",
                      gaussian_kernel_size: int = 5,
                      gaussian_sigma: float = 1.0) -> np.ndarray:
    """
    Phase I preprocessing pipeline:

        1. Gaussian smoothing
        2. CLAHE contrast normalization
        3. Color space conversion

    Returns processed image.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    # Step 1: Gaussian smoothing
    smoothed = apply_gaussian_filter(
        image,
        kernel_size=gaussian_kernel_size,
        sigma=gaussian_sigma
    )

    # Step 2: CLAHE
    enhanced = apply_clahe_lab(smoothed)

    # Step 3: Convert color space
    converted = convert_color_space(enhanced, color_space)

    if output_path:
        cv2.imwrite(output_path, converted)

    return converted