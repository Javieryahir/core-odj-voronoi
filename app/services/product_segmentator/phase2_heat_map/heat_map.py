import cv2
import numpy as np


# =========================================
# CONTRAST ANALYSIS
# =========================================

def analyze_contrast(gray: np.ndarray):
    """
    Returns mean and standard deviation of image.
    """
    mean = np.mean(gray)
    std = np.std(gray)
    return mean, std


# =========================================
# DYNAMIC CANNY
# =========================================

def dynamic_canny(gray: np.ndarray):
    """
    Adaptive Canny thresholds based on image contrast.
    """
    _, std = analyze_contrast(gray)

    # Avoid extremely small thresholds
    std = max(std, 10)

    low = int(0.5 * std)
    high = int(1.5 * std)

    edges = cv2.Canny(gray, low, high)
    return edges, low, high


def strict_canny(gray: np.ndarray):
    """
    Fixed thresholds (original behavior).
    """
    low, high = 30, 120
    edges = cv2.Canny(gray, low, high)
    return edges, low, high


# =========================================
# MULTI-CHANNEL EDGE DETECTION
# =========================================

def detect_edges(image: np.ndarray, adaptive: bool):
    """
    Edge detection with optional adaptive mode.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if adaptive:
        edges_l, low, high = dynamic_canny(l)
    else:
        edges_l, low, high = strict_canny(l)

    # Additional channels help with white-on-white separation
    edges_a = cv2.Canny(a, 20, 80)
    edges_b = cv2.Canny(b, 20, 80)

    combined = cv2.bitwise_or(edges_l, edges_a)
    combined = cv2.bitwise_or(combined, edges_b)

    # Morphological closing
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    return combined, low, high


# =========================================
# DISTANCE TRANSFORM
# =========================================

def compute_distance_transform(edges: np.ndarray):
    inverted = cv2.bitwise_not(edges)
    _, binary = cv2.threshold(inverted, 1, 255, cv2.THRESH_BINARY)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    return dist_transform


# =========================================
# DYNAMIC SEED EXTRACTION
# =========================================

def extract_seeds(distance_map: np.ndarray,
                  adaptive: bool,
                  gray: np.ndarray):
    """
    Adaptive seed thresholding based on contrast.
    """

    normalized = cv2.normalize(distance_map, None, 0, 1.0,
                               cv2.NORM_MINMAX)

    if adaptive:
        _, std = analyze_contrast(gray)

        # Low contrast → lower threshold ratio
        if std < 40:
            threshold_ratio = 0.15
        elif std < 70:
            threshold_ratio = 0.25
        else:
            threshold_ratio = 0.35
    else:
        threshold_ratio = 0.25

    _, seeds = cv2.threshold(normalized,
                             threshold_ratio,
                             1.0,
                             cv2.THRESH_BINARY)

    seeds = (seeds * 255).astype(np.uint8)

    # Slight dilation in adaptive mode
    kernel = np.ones((3, 3), np.uint8)
    iterations = 2 if adaptive else 1
    seeds = cv2.dilate(seeds, kernel, iterations=iterations)

    return seeds, threshold_ratio


# =========================================
# FULL PIPELINE
# =========================================

def phase2_generate_seeds(image: np.ndarray,
                          adaptive: bool = False,
                          debug_output_dir: str = None):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges, low, high = detect_edges(image, adaptive)

    distance_map = compute_distance_transform(edges)

    seeds, threshold_ratio = extract_seeds(distance_map,
                                           adaptive,
                                           gray)

    if debug_output_dir:
        cv2.imwrite(f"{debug_output_dir}/phase2_edges.png", edges)

        dist_vis = cv2.normalize(distance_map, None,
                                 0, 255,
                                 cv2.NORM_MINMAX)
        cv2.imwrite(f"{debug_output_dir}/phase2_distance.png",
                    dist_vis.astype(np.uint8))

        cv2.imwrite(f"{debug_output_dir}/phase2_seeds.png", seeds)

        print(f"[Phase II] Canny thresholds: low={low}, high={high}")
        print(f"[Phase II] Seed threshold ratio: {threshold_ratio}")
        print(f"[Phase II] Adaptive mode: {adaptive}")

    return edges, distance_map, seeds
