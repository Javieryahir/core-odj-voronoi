import cv2
import numpy as np

SEGMENT_REGISTRY = {}

def register_segment(name):
    def decorator(cls):
        SEGMENT_REGISTRY[name] = cls()
        return cls
    return decorator


class SegmentationStrategy:
    def segment(self, image, gradient):
        raise NotImplementedError


@register_segment("watershed")
class WatershedSeg(SegmentationStrategy):

    def segment(self, image, gradient):

        _, thresh = cv2.threshold(
            gradient, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        _, markers = cv2.threshold(
            dist, 0.3 * dist.max(), 255, 0
        )

        markers = np.uint8(markers)
        _, markers = cv2.connectedComponents(markers)
        markers = cv2.watershed(image, markers)

        # Remove watershed borders (-1)
        markers[markers < 0] = 0

        return markers


# @register_segment("connected")
# class ConnectedSeg(SegmentationStrategy):

#     def segment(self, image, gradient):
#         _, thresh = cv2.threshold(
#             gradient, 0, 255,
#             cv2.THRESH_BINARY + cv2.THRESH_OTSU
#         )
#         _, labels = cv2.connectedComponents(thresh)
#         return labels


@register_segment("voronoi")
class VoronoiSeg(SegmentationStrategy):

    def segment(self, image, gradient):
        _, thresh = cv2.threshold(
            gradient, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        _, labels = cv2.connectedComponents(
            (dist > 0).astype(np.uint8)
        )
        return labels
    