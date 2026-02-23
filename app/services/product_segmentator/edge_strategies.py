import cv2
import numpy as np

EDGE_REGISTRY = {}

def register_edge(name):
    def decorator(cls):
        EDGE_REGISTRY[name] = cls()
        return cls
    return decorator


class EdgeStrategy:
    def compute(self, gray):
        raise NotImplementedError


# @register_edge("canny")
# class CannyEdge(EdgeStrategy):

#     def compute(self, gray):
#         sigma = np.std(gray)
#         lower = int(max(0, 0.66 * sigma))
#         upper = int(min(255, 1.33 * sigma))
#         return cv2.Canny(gray, lower, upper)


@register_edge("laplacian")
class LaplacianEdge(EdgeStrategy):

    def compute(self, gray):

        # Step 1: Gaussian smoothing (IMPORTANT)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Step 2: Laplacian
        lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

        # Step 3: Absolute value
        lap = np.absolute(lap)

        # Step 4: Convert to uint8 safely
        lap = np.uint8(lap)

        return lap
    