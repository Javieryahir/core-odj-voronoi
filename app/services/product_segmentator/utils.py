import numpy as np
import cv2
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


# -------------------------------------------------
# Vectorized IoU helpers (avoid Python-level O(n²))
# -------------------------------------------------
def _boxes_to_arrays(boxes):
    """Convert list of (x, y, w, h, area) to NumPy column arrays."""
    arr = np.array(boxes, dtype=np.float64)          # (N, 5)
    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 0] + arr[:, 2]
    y2 = arr[:, 1] + arr[:, 3]
    area = arr[:, 4]
    return x1, y1, x2, y2, area


def _iou_matrix(boxes):
    """Return an (N, N) IoU matrix computed entirely in NumPy."""
    x1, y1, x2, y2, area = _boxes_to_arrays(boxes)

    # Broadcast pairwise intersection coordinates
    inter_x1 = np.maximum(x1[:, None], x1[None, :])
    inter_y1 = np.maximum(y1[:, None], y1[None, :])
    inter_x2 = np.minimum(x2[:, None], x2[None, :])
    inter_y2 = np.minimum(y2[:, None], y2[None, :])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union = area[:, None] + area[None, :] - inter_area
    union = np.where(union == 0, 1, union)  # avoid division by zero

    return inter_area / union

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
# IoU between two boxes (x, y, w, h, area)
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
        return 0.0

    return inter_area / union


# -------------------------------------------------
# 1a) Percentile-based area filtering
#     Removes boxes whose area is below the given
#     percentile of the distribution of all box areas.
# -------------------------------------------------
def percentile_filter(boxes, percentile=20):
    """
    Remove boxes whose area falls below `percentile` of all box areas.

    Args:
        boxes:      list of (x, y, w, h, area)
        percentile: float in [0, 100]; boxes below this percentile are dropped

    Returns:
        Filtered list of boxes.
    """
    if not boxes:
        return []

    areas = np.array([b[4] for b in boxes], dtype=float)
    threshold = np.percentile(areas, percentile)

    return [b for b in boxes if b[4] >= threshold]


# -------------------------------------------------
# 1b) Remove fully contained boxes
#     A box is removed if it is completely enclosed
#     within any other box (boundary-inclusive).
# -------------------------------------------------
def remove_contained(boxes):
    """
    Remove any box that is fully contained within another box.

    Containment is defined as:
        boxA.x1 >= boxB.x1  AND  boxA.y1 >= boxB.y1
        AND boxA.x2 <= boxB.x2  AND  boxA.y2 <= boxB.y2
    where A != B.

    Uses vectorized NumPy broadcasting for O(n²) comparisons.

    Returns:
        List with contained boxes removed.
    """
    if len(boxes) <= 1:
        return list(boxes)

    x1, y1, x2, y2, _ = _boxes_to_arrays(boxes)

    # (i, j): is box i contained in box j?
    contained = (
        (x1[:, None] >= x1[None, :]) &
        (y1[:, None] >= y1[None, :]) &
        (x2[:, None] <= x2[None, :]) &
        (y2[:, None] <= y2[None, :])
    )
    # A box trivially contains itself — ignore the diagonal
    np.fill_diagonal(contained, False)

    # A box is contained if ANY other box fully encloses it
    is_contained = contained.any(axis=1)

    return [b for b, drop in zip(boxes, is_contained) if not drop]


# -------------------------------------------------
# 1c) IoU-based iterative box merging
#     Merges pairs of boxes that overlap above
#     `iou_threshold` based on their *original*
#     extents (not growing merged extents), which
#     prevents weak-overlap cascade merging.
# -------------------------------------------------
def merge_boxes(boxes, iou_threshold=0.3):
    """
    Iteratively merge boxes that have IoU > iou_threshold.

    To prevent cascade merging (where A+B → AB grows to absorb C even
    though original A and C had IoU=0), each merge pass evaluates IoU
    only on the *input* boxes at the start of that pass.  The merged
    bounding rectangle is the minimal axis-aligned rectangle covering
    both boxes.  Passes repeat until no further merges occur.

    Uses a vectorized IoU matrix each pass for speed.

    Args:
        boxes:         list of (x, y, w, h, area)
        iou_threshold: minimum IoU to trigger a merge

    Returns:
        Merged list of boxes.
    """
    boxes = list(boxes)

    changed = True
    while changed:
        changed = False
        n = len(boxes)
        if n <= 1:
            break

        iou_mat = _iou_matrix(boxes)
        np.fill_diagonal(iou_mat, 0)           # ignore self

        used = np.zeros(n, dtype=bool)
        new_boxes = []

        for i in range(n):
            if used[i]:
                continue

            # Indices that overlap with i above threshold
            overlaps = np.where((~used) & (iou_mat[i] > iou_threshold))[0]
            # Include i itself
            group = np.concatenate(([i], overlaps))

            if len(group) > 1:
                arr = np.array([boxes[k] for k in group])
                x1 = arr[:, 0].min()
                y1 = arr[:, 1].min()
                x2 = (arr[:, 0] + arr[:, 2]).max()
                y2 = (arr[:, 1] + arr[:, 3]).max()
                w, h = int(x2 - x1), int(y2 - y1)
                new_boxes.append((int(x1), int(y1), w, h, w * h))
                used[group] = True
                changed = True
            else:
                new_boxes.append(boxes[i])
                used[i] = True

        boxes = new_boxes

    return boxes


# -------------------------------------------------
# 1d) Aspect-ratio outlier filter
#     Removes boxes whose aspect ratio (w/h) lies
#     more than `max_sigma` standard deviations from
#     the mean aspect ratio of all boxes.
# -------------------------------------------------
def aspect_ratio_filter(boxes, max_sigma=2.0):
    """
    Remove boxes whose aspect ratio is too far from the population mean.

    The aspect ratio of each box is computed as w / h.  Boxes whose
    aspect ratio differs from the mean by more than `max_sigma` standard
    deviations are discarded.

    Args:
        boxes:     list of (x, y, w, h, area)
        max_sigma: number of standard deviations to use as the cutoff

    Returns:
        Filtered list of boxes.
    """
    if len(boxes) < 2:
        return boxes

    ratios = np.array([b[2] / b[3] if b[3] != 0 else 0.0 for b in boxes],
                      dtype=np.float64)
    mean = ratios.mean()
    std = ratios.std()

    if std == 0:
        return boxes

    return [b for b, r in zip(boxes, ratios)
            if abs(r - mean) <= max_sigma * std]


# -------------------------------------------------
# 2) Remove duplicate crops using IoU similarity
#    Boxes with IoU > duplicate_threshold are
#    considered the same product; keep the larger.
# -------------------------------------------------
def remove_duplicate_boxes(boxes, duplicate_threshold=0.7):
    """
    After all merging/cleaning, remove boxes that are near-duplicates of
    each other (IoU > duplicate_threshold).  When two duplicates are found
    the larger box (by area) is retained.

    Uses a vectorized IoU matrix for the heavy O(n²) comparison.

    Args:
        boxes:               list of (x, y, w, h, area)
        duplicate_threshold: IoU threshold above which two boxes are
                             considered duplicates (default 0.7)

    Returns:
        De-duplicated list of boxes.
    """
    if not boxes:
        return []

    # Sort largest-first so that when duplicates are found we keep the
    # largest by construction (first encountered wins).
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)

    n = len(boxes)
    if n == 1:
        return boxes

    iou_mat = _iou_matrix(boxes)
    np.fill_diagonal(iou_mat, 0)

    suppressed = np.zeros(n, dtype=bool)
    kept = []

    for i in range(n):
        if suppressed[i]:
            continue
        kept.append(boxes[i])
        # Suppress every smaller box that is a duplicate of i
        dups = np.where((~suppressed) & (iou_mat[i] > duplicate_threshold))[0]
        suppressed[dups] = True

    return kept


# -------------------------------------------------
# 3) Diagonal-size grouping
#
#    Shared clustering core + two public interfaces:
#      * filter_by_diagonal_groups  - returns a flat list (used by the
#                                     draw / postprocess pipeline)
#      * get_diagonal_groups        - returns labelled groups with a
#                                     human-readable subfolder name each
#                                     (used by the crop-saving step)
# -------------------------------------------------

def _build_diagonal_groups(boxes, gap_ratio):
    """
    Private helper: sort boxes by diagonal and split into clusters
    wherever consecutive diagonals differ by more than gap_ratio.

    Returns a list of lists -- each inner list is a raw group of
    (x, y, w, h, area) boxes.
    """
    diag_boxes = sorted(
        ((np.sqrt(b[2] ** 2 + b[3] ** 2), b) for b in boxes),
        key=lambda t: t[0],
    )

    groups = []
    current_group = [diag_boxes[0][1]]

    for i in range(1, len(diag_boxes)):
        prev_diag = diag_boxes[i - 1][0]
        curr_diag  = diag_boxes[i][0]
        relative_gap = (curr_diag - prev_diag) / prev_diag if prev_diag > 0 else 0.0

        if relative_gap > gap_ratio:
            groups.append(current_group)
            current_group = []

        current_group.append(diag_boxes[i][1])

    groups.append(current_group)
    return groups


def filter_by_diagonal_groups(boxes, gap_ratio=0.3, min_group_size=2):
    """
    Group boxes by diagonal size and discard groups that are too small.

    Algorithm
    ---------
    1. Compute diagonal = sqrt(w^2 + h^2) for every box.
    2. Sort boxes by diagonal (ascending).
    3. Walk the sorted list; start a new group whenever the relative
       gap between two consecutive diagonals exceeds gap_ratio.
    4. Discard every group whose member count < min_group_size.

    Args:
        boxes:          list of (x, y, w, h, area)
        gap_ratio:      relative diagonal jump that starts a new group.
        min_group_size: groups smaller than this are discarded.

    Returns:
        Filtered flat list of boxes (order may differ from input).
    """
    if not boxes:
        return []

    kept = []
    for group in _build_diagonal_groups(boxes, gap_ratio):
        if len(group) >= min_group_size:
            kept.extend(group)

    return kept


def get_diagonal_groups(boxes, gap_ratio=0.3, min_group_size=2):
    """
    Same clustering as filter_by_diagonal_groups, but returns the groups
    individually so callers can organise output files into per-group
    subfolders.

    The subfolder name for each group is derived from the average width and
    height of its members:

        avg_{mean_w}x{mean_h}px   e.g. avg_142x198px

    Args:
        boxes:          list of (x, y, w, h, area) -- already post-processed
        gap_ratio:      same semantics as filter_by_diagonal_groups
        min_group_size: groups smaller than this are still discarded

    Returns:
        list of (subfolder_name: str, group_boxes: list)
        Sorted by ascending average diagonal (smallest group first).
    """
    if not boxes:
        return []

    result = []
    for group in _build_diagonal_groups(boxes, gap_ratio):
        if len(group) < min_group_size:
            continue

        mean_w = int(round(np.mean([b[2] for b in group])))
        mean_h = int(round(np.mean([b[3] for b in group])))
        subfolder = f"avg_{mean_w}x{mean_h}px"

        result.append((subfolder, group))

    return result


# -------------------------------------------------
# Full post-processing pipeline
# -------------------------------------------------
def postprocess_boxes(
    boxes,
    area_percentile=20,
    iou_merge_threshold=0.3,
    aspect_ratio_sigma=2.0,
    duplicate_threshold=0.7,
    diagonal_gap_ratio=0.3,
    min_group_size=2,
):
    """
    Apply the full Phase IV post-processing chain in order:
        1a. Percentile-based area filtering
        1b. Remove fully contained boxes
        1c. Iterative IoU merging
        1d. Aspect-ratio outlier filtering
         2. Remove near-duplicate boxes
         3. Diagonal-size grouping filter

    All thresholds are configurable.
    """
    boxes = percentile_filter(boxes, percentile=area_percentile)
    boxes = remove_contained(boxes)
    boxes = merge_boxes(boxes, iou_threshold=iou_merge_threshold)
    boxes = aspect_ratio_filter(boxes, max_sigma=aspect_ratio_sigma)
    boxes = remove_duplicate_boxes(boxes, duplicate_threshold=duplicate_threshold)
    boxes = filter_by_diagonal_groups(
        boxes,
        gap_ratio=diagonal_gap_ratio,
        min_group_size=min_group_size,
    )
    return boxes


# -------------------------------------------------
# Final Phase IV draw
# -------------------------------------------------
def draw_bounding_boxes(
    image,
    labels,
    disable_filtering=False,
    area_percentile=20,
    iou_merge_threshold=0.3,
    aspect_ratio_sigma=2.0,
    duplicate_threshold=0.7,
    diagonal_gap_ratio=0.3,
    min_group_size=2,
):
    output = image.copy()
    boxes = extract_boxes(labels)

    if not disable_filtering:
        boxes = postprocess_boxes(
            boxes,
            area_percentile=area_percentile,
            iou_merge_threshold=iou_merge_threshold,
            aspect_ratio_sigma=aspect_ratio_sigma,
            duplicate_threshold=duplicate_threshold,
            diagonal_gap_ratio=diagonal_gap_ratio,
            min_group_size=min_group_size,
        )

    for box in boxes:
        x, y, w, h, _ = box
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

    return output


# -------------------------------------------------
# Jaccard similarity (pixel mask IoU)
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
        return 0.0

    return intersection / union


# -------------------------------------------------
# Load ground truth images (for Jaccard matching)
# -------------------------------------------------
def load_ground_truth(folder="results"):

    images = {}

    if not os.path.exists(folder):
        return images

    for file in os.listdir(folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            images[file] = cv2.imread(os.path.join(folder, file))

    return images


# -------------------------------------------------
# Load ground truth bounding boxes for evaluation
#
# Supported formats
#   JSON  – list of {"x":…,"y":…,"w":…,"h":…} or [x,y,w,h]
#   TXT   – one box per line: x y w h
# -------------------------------------------------
def load_gt_boxes(path):
    """
    Load ground-truth bounding boxes from a file.

    Supported formats
    -----------------
    JSON  :  A JSON array of objects {"x", "y", "w", "h"}
             or plain arrays [x, y, w, h].
    TXT   :  One box per line with whitespace-separated x y w h values.

    Returns
    -------
    list of (x, y, w, h, area) tuples
    """
    ext = os.path.splitext(path)[1].lower()
    boxes = []

    if ext == ".json":
        with open(path) as f:
            data = json.load(f)
        for item in data:
            if isinstance(item, dict):
                x, y, w, h = item["x"], item["y"], item["w"], item["h"]
            else:
                x, y, w, h = item[:4]
            boxes.append((int(x), int(y), int(w), int(h), int(w) * int(h)))

    elif ext == ".txt":
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    boxes.append((x, y, w, h, w * h))

    else:
        raise ValueError(f"Unsupported ground-truth format: {ext}  (use .json or .txt)")

    return boxes


# -------------------------------------------------
# Evaluation metrics
# -------------------------------------------------
def evaluate_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Compare predicted boxes against ground-truth boxes and compute:
        Precision, Recall, F1-score, Mean IoU (over true positives).

    A predicted box is a True Positive (TP) if its best IoU with any
    unmatched GT box exceeds `iou_threshold`.

    Args:
        pred_boxes:    list of (x, y, w, h, area)
        gt_boxes:      list of (x, y, w, h, area)
        iou_threshold: IoU required to count as a match (default 0.5)

    Returns:
        dict with keys: precision, recall, f1, mean_iou, tp, fp, fn
    """
    tp = 0
    fp = 0
    matched_gt = set()
    iou_scores = []

    for pred in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1

        for g_idx, gt in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
            iou_scores.append(best_iou)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mean_iou  = float(np.mean(iou_scores)) if iou_scores else 0.0

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "mean_iou":  mean_iou,
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
    }
