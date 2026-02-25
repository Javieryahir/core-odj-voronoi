import os
import cv2
from multiprocessing import Pool, cpu_count
from .edge_strategies import EDGE_REGISTRY
from .segmentation_strategies import SEGMENT_REGISTRY
from .utils import (
    colorize_labels,
    draw_bounding_boxes,
    extract_boxes,
    postprocess_boxes,
    get_diagonal_groups,
    load_gt_boxes,
    evaluate_boxes,
)


# -------------------------------------------------
# Worker
# -------------------------------------------------
def run_combination(args):
    (
        image,
        edge_name,
        seg_name,
        output_root,
        disable_box_filtering,
        area_percentile,
        iou_merge_threshold,
        aspect_ratio_sigma,
        duplicate_threshold,
        diagonal_gap_ratio,
        min_group_size,
    ) = args

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edge = EDGE_REGISTRY[edge_name]
    segmenter = SEGMENT_REGISTRY[seg_name]

    gradient = edge.compute(gray)
    labels = segmenter.segment(image.copy(), gradient)

    folder = os.path.join(output_root, f"{edge_name}_{seg_name}")
    os.makedirs(folder, exist_ok=True)

    # Save gradient
    cv2.imwrite(os.path.join(folder, "gradient.png"), gradient)

    # Color visualization
    colored = colorize_labels(labels)
    cv2.imwrite(os.path.join(folder, "labels_color.png"), colored)

    overlay = cv2.addWeighted(image, 0.6, colored, 0.4, 0)
    cv2.imwrite(os.path.join(folder, "overlay.png"), overlay)

    # Bounding-box image
    boxed = draw_bounding_boxes(
        image,
        labels,
        disable_filtering=disable_box_filtering,
        area_percentile=area_percentile,
        iou_merge_threshold=iou_merge_threshold,
        aspect_ratio_sigma=aspect_ratio_sigma,
        duplicate_threshold=duplicate_threshold,
        diagonal_gap_ratio=diagonal_gap_ratio,
        min_group_size=min_group_size,
    )
    cv2.imwrite(os.path.join(folder, "boxes.png"), boxed)

    # -------------------------------------------------
    # Crop + save product images, organised by size group
    # -------------------------------------------------
    result_folder = os.path.join(output_root, "result-images")
    os.makedirs(result_folder, exist_ok=True)

    # Obtain final post-processed boxes (same pipeline used for drawing)
    raw_boxes = extract_boxes(labels)
    if not disable_box_filtering:
        final_boxes = postprocess_boxes(
            raw_boxes,
            area_percentile=area_percentile,
            iou_merge_threshold=iou_merge_threshold,
            aspect_ratio_sigma=aspect_ratio_sigma,
            duplicate_threshold=duplicate_threshold,
            diagonal_gap_ratio=diagonal_gap_ratio,
            min_group_size=min_group_size,
        )
    else:
        final_boxes = raw_boxes

    # Group surviving boxes by diagonal size so each group gets its own
    # subfolder named after the average dimensions of that group.
    # When filtering is disabled we still group, but with min_group_size=1
    # so no boxes are discarded at this stage.
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
            filename = f"{edge_name}_{seg_name}_crop_{i:04d}.png"
            cv2.imwrite(os.path.join(group_folder, filename), crop)

    print(f"Finished: {edge_name} + {seg_name}")

    # Return final boxes so the main process can run evaluation
    return (edge_name, seg_name, final_boxes)


# -------------------------------------------------
# Main parallel entry point
# -------------------------------------------------
def run_parallel(
    image,
    output_root,
    disable_box_filtering=False,
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
    ]

    args_list = [
        (
            image,
            edge,
            seg,
            output_root,
            disable_box_filtering,
            area_percentile,
            iou_merge_threshold,
            aspect_ratio_sigma,
            duplicate_threshold,
            diagonal_gap_ratio,
            min_group_size,
        )
        for edge, seg in combinations
    ]

    if not output_root:
        raise ValueError("'output_root' was not specified")

    workers = min(len(combinations), cpu_count())
    print(f"Running in parallel using {workers} workers")

    with Pool(processes=workers) as pool:
        results = pool.map(run_combination, args_list)

    # ── Evaluation ─────────────────────────────────────────────────────────
    if evaluate and gt_path:
        _run_evaluation(results, gt_path, eval_iou_threshold)


def _run_evaluation(results, gt_path, eval_iou_threshold):
    """
    Load GT boxes from gt_path and compute evaluation metrics for every
    (edge, segmentation) combination that was processed.
    """
    print("\n" + "=" * 60)
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

    print(f"  Ground-truth boxes: {len(gt_boxes)}\n")

    all_metrics = []

    for edge_name, seg_name, pred_boxes in results:
        combo = f"{edge_name} + {seg_name}"
        metrics = evaluate_boxes(pred_boxes, gt_boxes, iou_threshold=eval_iou_threshold)
        all_metrics.append(metrics)

        print(f"  [{combo}]")
        print(f"    Predicted boxes : {len(pred_boxes)}")
        print(f"    TP / FP / FN    : {metrics['tp']} / {metrics['fp']} / {metrics['fn']}")
        print(f"    Precision       : {metrics['precision']:.4f}")
        print(f"    Recall          : {metrics['recall']:.4f}")
        print(f"    F1-score        : {metrics['f1']:.4f}")
        print(f"    Mean IoU (TPs)  : {metrics['mean_iou']:.4f}")
        print()

    # Aggregate over all combinations
    if len(all_metrics) > 1:
        import numpy as np
        print("  [Aggregate across all combinations]")
        for key in ("precision", "recall", "f1", "mean_iou"):
            values = [m[key] for m in all_metrics]
            print(f"    Mean {key:<12}: {float(np.mean(values)):.4f}")
        print()

    print("=" * 60)
