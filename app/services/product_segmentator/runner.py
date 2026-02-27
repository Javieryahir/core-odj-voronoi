import os
import cv2
from multiprocessing import Pool, cpu_count
from .edge_strategies import EDGE_REGISTRY
from .segmentation_strategies import SEGMENT_REGISTRY
from .utils import (
    colorize_labels,
    draw_boxes_on_image,
    extract_boxes,
    postprocess_boxes,
    get_diagonal_groups,
    load_gt_boxes,
    evaluate_boxes,
)


# -------------------------------------------------
# Worker – segmentation + raw box extraction only
# -------------------------------------------------
def run_combination(args):
    (
        image,
        edge_name,
        seg_name,
        output_root,
    ) = args

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edge = EDGE_REGISTRY[edge_name]
    segmenter = SEGMENT_REGISTRY[seg_name]

    gradient = edge.compute(gray)
    labels = segmenter.segment(image.copy(), gradient)

    # Save per-method debug visuals
    folder = os.path.join(output_root, f"{edge_name}_{seg_name}")
    os.makedirs(folder, exist_ok=True)

    cv2.imwrite(os.path.join(folder, "gradient.png"), gradient)

    colored = colorize_labels(labels)
    cv2.imwrite(os.path.join(folder, "labels_color.png"), colored)

    overlay = cv2.addWeighted(image, 0.6, colored, 0.4, 0)
    cv2.imwrite(os.path.join(folder, "overlay.png"), overlay)

    # Extract raw boxes – no filtering here
    raw_boxes = extract_boxes(labels)

    print(f"Finished: {edge_name} + {seg_name}  ({len(raw_boxes)} raw boxes)")

    return (edge_name, seg_name, raw_boxes)


# -------------------------------------------------
# Main parallel entry point
# -------------------------------------------------
def run_parallel(
    image,
    output_root="output",
    disable_box_filtering=False,
    max_box_ratio=0.9,
    area_percentile=20.0,
    iou_merge_threshold=0.3,
    aspect_ratio_sigma=2.0,
    duplicate_threshold=0.7,
    diagonal_gap_ratio=0.3,
    min_group_size=2,
    evaluate=False,
    gt_path=None,
    eval_iou_threshold=0.5,
):
    combinations = [
        ("laplacian", "watershed"),
        ("laplacian", "convex_hull"),
        ("laplacian", "alpha_shape"),
        ("laplacian", "dist_transform"),
        ("laplacian", "region_growing"),
        # ("laplacian", "random_walker"),
        ("laplacian", "split_merge"),
        ("laplacian", "mser"),
        ("laplacian", "kmeans"),
    ]

    # Only pass what workers need (image + names + output path)
    args_list = [
        (image, edge, seg, output_root)
        for edge, seg in combinations
    ]

    workers = min(len(combinations), cpu_count())
    print(f"Running {len(combinations)} combinations in parallel using {workers} workers")

    with Pool(processes=workers) as pool:
        results = pool.map(run_combination, args_list)

    # -------------------------------------------------
    # Aggregate raw boxes from ALL methods
    # -------------------------------------------------
    all_raw_boxes = []
    for edge_name, seg_name, raw_boxes in results:
        all_raw_boxes.extend(raw_boxes)

    print(f"\nAggregated {len(all_raw_boxes)} raw boxes from {len(results)} methods")

    # -------------------------------------------------
    # Global filtering on the aggregated set
    # -------------------------------------------------
    image_shape = image.shape[:2]

    if not disable_box_filtering:
        final_boxes = postprocess_boxes(
            all_raw_boxes,
            image_shape=image_shape,
            max_box_ratio=max_box_ratio,
            area_percentile=area_percentile,
            iou_merge_threshold=iou_merge_threshold,
            aspect_ratio_sigma=aspect_ratio_sigma,
            duplicate_threshold=duplicate_threshold,
            diagonal_gap_ratio=diagonal_gap_ratio,
            min_group_size=min_group_size,
        )
    else:
        final_boxes = all_raw_boxes

    print(f"After global filtering: {len(final_boxes)} boxes\n")

    # -------------------------------------------------
    # Draw global bounding-box image
    # -------------------------------------------------
    boxed = draw_boxes_on_image(image, final_boxes)
    cv2.imwrite(os.path.join(output_root, "boxes.png"), boxed)

    # -------------------------------------------------
    # Crop + save product images, organised by size group
    # -------------------------------------------------
    result_folder = os.path.join(output_root, "result-images")
    os.makedirs(result_folder, exist_ok=True)

    effective_min = min_group_size if not disable_box_filtering else 1
    groups = get_diagonal_groups(
        final_boxes,
        gap_ratio=diagonal_gap_ratio,
        min_group_size=effective_min,
    )

    for subfolder_name, group_boxes in groups:
        group_folder = os.path.join(result_folder, subfolder_name)
        os.makedirs(group_folder, exist_ok=True)

        for i, (x, y, w, h, area) in enumerate(group_boxes):
            crop = image[y:y + h, x:x + w]
            filename = f"crop_{i:04d}.png"
            cv2.imwrite(os.path.join(group_folder, filename), crop)

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------
    if evaluate and gt_path:
        _run_evaluation(final_boxes, gt_path, eval_iou_threshold)


def _run_evaluation(pred_boxes, gt_path, eval_iou_threshold):
    """
    Compute evaluation metrics for the globally-filtered boxes.
    """
    print("=" * 60)
    print("EVALUATION RESULTS")
    print(f"  Ground-truth file : {gt_path}")
    print(f"  IoU threshold     : {eval_iou_threshold}")
    print("=" * 60)

    try:
        gt_boxes = load_gt_boxes(gt_path)
    except Exception as e:
        print(f"[ERROR] Could not load ground-truth boxes: {e}")
        return

    if not gt_boxes:
        print("[WARNING] Ground-truth file loaded but contains no boxes.")
        return

    print(f"  Ground-truth boxes: {len(gt_boxes)}")
    print(f"  Predicted boxes   : {len(pred_boxes)}\n")

    metrics = evaluate_boxes(pred_boxes, gt_boxes, iou_threshold=eval_iou_threshold)

    print(f"  TP / FP / FN  : {metrics['tp']} / {metrics['fp']} / {metrics['fn']}")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print(f"  F1-score      : {metrics['f1']:.4f}")
    print(f"  Mean IoU (TPs): {metrics['mean_iou']:.4f}")
    print()
    print("=" * 60)
