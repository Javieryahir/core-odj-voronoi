import cv2
import numpy as np


def extract_bounding_boxes(markers: np.ndarray,
                           min_area: int = 500,
                           rotated: bool = False):
    """
    Extract bounding boxes from labeled marker map.

    Parameters:
        markers: int32 label map
        min_area: filter small noise segments
        rotated: use minAreaRect if True

    Returns:
        List of bounding boxes
    """

    boxes = []
    unique_labels = np.unique(markers)

    for label in unique_labels:
        if label <= 0:
            continue  # skip background and watershed borders

        mask = (markers == label).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        if rotated:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            boxes.append(("rotated", box))
        else:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append(("axis", (x, y, w, h)))

    return boxes


def draw_bounding_boxes(image: np.ndarray,
                        boxes,
                        color=(0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes on image.
    """
    output = image.copy()

    for box_type, box in boxes:
        if box_type == "axis":
            x, y, w, h = box
            cv2.rectangle(output,
                          (x, y),
                          (x + w, y + h),
                          color,
                          thickness)

        elif box_type == "rotated":
            cv2.drawContours(output,
                             [box],
                             0,
                             color,
                             thickness)

    return output


def phase4_rectanglize(original_image: np.ndarray,
                       markers: np.ndarray,
                       debug_output_dir: str = None,
                       rotated: bool = False):
    """
    Full Phase IV pipeline:
        1. Extract bounding boxes
        2. Draw them
        3. Save final output
    """

    boxes = extract_bounding_boxes(markers, rotated=rotated)

    output_image = draw_bounding_boxes(original_image, boxes)

    if debug_output_dir:
        cv2.imwrite(f"{debug_output_dir}/phase4_bounding_boxes.png",
                    output_image)

    return boxes, output_image
