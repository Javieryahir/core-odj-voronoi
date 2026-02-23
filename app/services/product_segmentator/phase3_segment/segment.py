import cv2
import numpy as np


# ==========================
# WATERSHED IMPLEMENTATION
# ==========================

def create_markers_from_seeds(seeds: np.ndarray) -> np.ndarray:
    num_labels, markers = cv2.connectedComponents(seeds)
    return markers.astype(np.int32)


def compute_gradient_image(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    gradient = cv2.magnitude(grad_x, grad_y)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)
    return gradient.astype(np.uint8)


def watershed_segmentation(image: np.ndarray,
                            seeds: np.ndarray,
                            debug_output_dir: str = None) -> np.ndarray:

    markers = create_markers_from_seeds(seeds)

    image_ws = image.copy()
    if len(image_ws.shape) != 3:
        image_ws = cv2.cvtColor(image_ws, cv2.COLOR_GRAY2BGR)

    cv2.watershed(image_ws, markers)

    if debug_output_dir:
        boundary_vis = image.copy()
        boundary_vis[markers == -1] = [0, 0, 255]
        cv2.imwrite(f"{debug_output_dir}/phase3_watershed_boundaries.png",
                    boundary_vis)

    return markers


# ==========================
# CANNY-BASED SEGMENTATION
# ==========================

def canny_segmentation(image: np.ndarray,
                       debug_output_dir: str = None) -> np.ndarray:
    """
    Segment objects using Canny edges + contour filling.
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Canny
    edges = cv2.Canny(gray, 50, 150)

    # Close gaps between edges
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Create label map
    markers = np.zeros(gray.shape, dtype=np.int32)

    for idx, contour in enumerate(contours):
        cv2.drawContours(markers, [contour], -1, idx + 1, -1)

    if debug_output_dir:
        cv2.imwrite(f"{debug_output_dir}/phase3_canny_edges.png", edges)
        cv2.imwrite(f"{debug_output_dir}/phase3_canny_closed.png", closed)

        vis = image.copy()
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(f"{debug_output_dir}/phase3_canny_contours.png", vis)

    return markers


# ==========================
# CONTROLLER
# ==========================

def phase3_segment(image: np.ndarray,
                   seeds: np.ndarray,
                   method: str,
                   debug_output_dir: str = None) -> np.ndarray:

    if method == "watershed":
        return watershed_segmentation(image, seeds, debug_output_dir)

    elif method == "canny":
        return canny_segmentation(image, debug_output_dir)

    else:
        raise ValueError("Method must be 'watershed' or 'canny'")
    