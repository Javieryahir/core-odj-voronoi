import numpy as np
import cv2
import os


# -------------------------------------------------
# Colorize labels
# -------------------------------------------------
def colorize_labels(labels):
    np.random.seed(42)
    h, w = labels.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for label in np.unique(labels):
        if label == 0:
            continue
        color = np.random.randint(0, 255, size=3)
        colored[labels == label] = color

    return colored


# -------------------------------------------------
# Extract raw bounding boxes
# -------------------------------------------------
def extract_boxes(labels):

    boxes = []

    for label in np.unique(labels):
        if label == 0:
            continue

        mask = (labels == label).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h, w * h))

    return boxes


# -------------------------------------------------
# IoU
# -------------------------------------------------
def compute_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    union = boxA[4] + boxB[4] - inter_area
    if union == 0:
        return 0

    return inter_area / union


# -------------------------------------------------
# Merge (IoU only – controlled)
# -------------------------------------------------
def merge_boxes(boxes, iou_threshold=0.3):

    boxes = list(boxes)
    merged = True

    while merged:
        merged = False
        new_boxes = []
        used = set()

        for i in range(len(boxes)):
            if i in used:
                continue

            boxA = boxes[i]
            merged_box = boxA

            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue

                boxB = boxes[j]

                if compute_iou(merged_box, boxB) > iou_threshold:

                    x = min(merged_box[0], boxB[0])
                    y = min(merged_box[1], boxB[1])
                    w = max(
                        merged_box[0] + merged_box[2],
                        boxB[0] + boxB[2]
                    ) - x
                    h = max(
                        merged_box[1] + merged_box[3],
                        boxB[1] + boxB[3]
                    ) - y

                    merged_box = (x, y, w, h, w * h)

                    used.add(j)
                    merged = True

            new_boxes.append(merged_box)

        boxes = new_boxes

    return boxes


# -------------------------------------------------
# Remove contained
# -------------------------------------------------
def remove_contained(boxes):

    result = []

    for i, boxA in enumerate(boxes):

        contained = False

        for j, boxB in enumerate(boxes):
            if i == j:
                continue

            if (
                boxA[0] >= boxB[0] and
                boxA[1] >= boxB[1] and
                boxA[0] + boxA[2] <= boxB[0] + boxB[2] and
                boxA[1] + boxA[3] <= boxB[1] + boxB[3]
            ):
                contained = True
                break

        if not contained:
            result.append(boxA)

    return result


# -------------------------------------------------
# Percentile filtering
# -------------------------------------------------
def percentile_filter(boxes, percentile=60):

    if not boxes:
        return []

    areas = np.array([b[4] for b in boxes])
    threshold = np.percentile(areas, percentile)

    return [b for b in boxes if b[4] >= threshold]


# -------------------------------------------------
# Final Phase IV
# -------------------------------------------------
def draw_bounding_boxes(image, labels, disable_filtering=False):

    output = image.copy()
    boxes = extract_boxes(labels)

    if not disable_filtering:
        boxes = merge_boxes(boxes)
        boxes = remove_contained(boxes)
        boxes = percentile_filter(boxes, percentile=90)

    for box in boxes:
        x, y, w, h, _ = box
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    return output

# -------------------------------------------------
# Jaccard similarity (IoU on masks)
# -------------------------------------------------
def compute_jaccard(img1, img2):

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    _, bin1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
    _, bin2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)

    bin1 = bin1 > 0
    bin2 = bin2 > 0

    intersection = np.logical_and(bin1, bin2).sum()
    union = np.logical_or(bin1, bin2).sum()

    if union == 0:
        return 0

    return intersection / union


# -------------------------------------------------
# Load ground truth images
# -------------------------------------------------
def load_ground_truth(folder):

    images = {}

    if not os.path.exists(folder):
        return images

    for file in os.listdir(folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            images[file] = cv2.imread(os.path.join(folder, file))

    return images
