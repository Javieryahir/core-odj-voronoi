import os
import cv2
from multiprocessing import Pool, cpu_count, Manager
from .edge_strategies import EDGE_REGISTRY
from .segmentation_strategies import SEGMENT_REGISTRY
from .utils import (
    colorize_labels,
    draw_bounding_boxes,
    compute_jaccard,
    load_ground_truth,
)


CWD = os.path.join(os.getcwd(), "tmp")

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
        gt_images,
        best_scores,
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

    boxed = draw_bounding_boxes(
        image,
        labels,
        disable_filtering=disable_box_filtering
    )

    cv2.imwrite(os.path.join(folder, "boxes.png"), boxed)

    # -------------------------------------------------
    # EVALUATION PART
    # -------------------------------------------------
    result_folder = os.path.join(output_root, "result-images")
    os.makedirs(result_folder, exist_ok=True)

    # Extract boxes again
    from .utils import extract_boxes
    boxes = extract_boxes(labels)
    MIN_AREA = 500

    for (x, y, w, h, area) in boxes:

        if area < MIN_AREA:
            continue

        crop = image[y:y+h, x:x+w]

        for name, gt in gt_images.items():

            gt_area = gt.shape[0] * gt.shape[1]
            size_ratio = min(area, gt_area) / max(area, gt_area)

            if size_ratio < 0.3:
                continue

            score = compute_jaccard(crop, gt)

            if score > best_scores.get(name, 0):
                best_scores[name] = score
                cv2.imwrite(os.path.join(result_folder, name), crop)

                print(
                    f"Updated best for {name}: "
                    f"{score:.3f} "
                    f"({edge_name}+{seg_name})"
                )

    print(f"Finished: {edge_name} + {seg_name}")


# -------------------------------------------------
# Main parallel
# -------------------------------------------------
def run_parallel(image,
                 output_root,
                 disable_box_filtering=False):

    combinations = [
        ("laplacian", "watershed"),
        ("laplacian", "voronoi"),
    ]

    gt_images = load_ground_truth(os.path.join(CWD, "dev_results"))

    manager = Manager()
    best_scores = manager.dict()

    args_list = [
        (
            image,
            edge,
            seg,
            output_root,
            disable_box_filtering,
            gt_images,
            best_scores,
        )
        for edge, seg in combinations
    ]

    workers = min(len(combinations), cpu_count())

    print(f"Running in parallel using {workers} workers")

    with Pool(processes=workers) as pool:
        pool.map(run_combination, args_list)

    # Print final summary
    print("\nFinal Best Scores:")
    for name, score in best_scores.items():
        print(f"{name}: {score:.3f}")
