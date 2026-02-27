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


# =====================================================================
# Shared helpers
# =====================================================================

def _gradient_to_interior_mask(gradient, close_kernel=5, close_iter=2,
                               open_kernel=3, open_iter=1):
    """
    Common preprocessing used by multiple strategies:
        gradient → Otsu edges → close gaps → invert → open bridges
    Returns a binary uint8 mask where 255 = object interior, 0 = edge/bg.
    """
    _, thresh = cv2.threshold(
        gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
    )
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close,
                              iterations=close_iter)
    inverted = cv2.bitwise_not(closed)

    if open_kernel > 0 and open_iter > 0:
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)
        )
        inverted = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, k_open,
                                    iterations=open_iter)
    return inverted


def _interior_connected_components(gradient, min_ratio=0.003, **kw):
    """
    Compute connected components on the interior mask derived from the
    gradient. Returns (n_labels, cc_labels, img_area, min_area).
    Labels 0 is always background.
    """
    interior = _gradient_to_interior_mask(gradient, **kw)
    n_labels, cc_labels = cv2.connectedComponents(interior)
    img_area = gradient.shape[0] * gradient.shape[1]
    min_area = img_area * min_ratio
    return n_labels, cc_labels, img_area, min_area


def _filter_labels_by_area(labels, n_labels, img_area, min_area,
                           max_ratio=0.5):
    """
    Zero out labels that are too small or too large (background).
    Returns cleaned label map (int32, 0-based background).
    """
    out = labels.copy().astype(np.int32)
    for lbl in range(1, n_labels):
        count = int(np.sum(labels == lbl))
        if count < min_area or count > max_ratio * img_area:
            out[out == lbl] = 0
    return out


# =====================================================================
# 1. Watershed (original, kept from user's file)
# =====================================================================

@register_segment("watershed")
class WatershedSeg(SegmentationStrategy):

    def segment(self, image, gradient):

        _, thresh = cv2.threshold(
            gradient, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,
                                  iterations=2)
        sure_bg = cv2.dilate(opened, kernel, iterations=3)

        dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(image, markers)
        markers[markers < 0] = 0

        return markers


# =====================================================================
# 2. Convex Hull (kept from previous work)
# =====================================================================

@register_segment("convex_hull")
class ConvexHullSeg(SegmentationStrategy):
    """
    Find enclosed interior regions via connected components on the
    inverted edge map, then wrap each in a convex hull.
    """

    def segment(self, image, gradient):

        n_labels, cc_labels, img_area, min_area = \
            _interior_connected_components(gradient)

        if n_labels <= 1:
            return np.zeros(gradient.shape, dtype=np.int32)

        hulls = []
        for lbl in range(1, n_labels):
            mask = (cc_labels == lbl).astype(np.uint8)
            if mask.sum() < min_area:
                continue

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            hull = cv2.convexHull(np.vstack(contours))
            hull_area = cv2.contourArea(hull)
            if hull_area < min_area:
                continue

            hulls.append((hull_area, hull))

        if not hulls:
            return np.zeros(gradient.shape, dtype=np.int32)

        hulls = [(a, h) for a, h in hulls if a < 0.5 * img_area]
        hulls.sort(key=lambda t: t[0], reverse=True)

        labels = np.zeros(gradient.shape, dtype=np.int32)
        for idx, (_, hull) in enumerate(hulls, start=1):
            cv2.drawContours(labels, [hull], -1, int(idx),
                             thickness=cv2.FILLED)
        return labels


# =====================================================================
# 3. Alpha Shapes (concave hull)
#    Uses Scipy Delaunay triangulation filtered by circumradius.
# =====================================================================

@register_segment("alpha_shape")
class AlphaShapeSeg(SegmentationStrategy):
    """
    Concave hull via alpha shapes.  For each interior region:
        1. Extract boundary points.
        2. Delaunay triangulate.
        3. Keep only triangles whose circumradius < 1/alpha.
        4. Extract and fill the boundary polygon.
    """

    @staticmethod
    def _circumradius(tri_pts):
        """Circumradius of a triangle defined by 3 points (2D)."""
        a = np.linalg.norm(tri_pts[0] - tri_pts[1])
        b = np.linalg.norm(tri_pts[1] - tri_pts[2])
        c = np.linalg.norm(tri_pts[2] - tri_pts[0])
        s = (a + b + c) / 2.0
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
        if area == 0:
            return np.inf
        return (a * b * c) / (4.0 * area)

    @staticmethod
    def _alpha_shape_polygon(points, alpha):
        """
        Compute the alpha-shape boundary from a set of 2D points.
        Returns a list of boundary points forming the concave hull,
        or None if degenerate.
        """
        from scipy.spatial import Delaunay

        if len(points) < 4:
            return cv2.convexHull(points.astype(np.float32))

        try:
            tri = Delaunay(points)
        except Exception:
            return cv2.convexHull(points.astype(np.float32))

        # Filter triangles by circumradius
        max_r = 1.0 / alpha if alpha > 0 else np.inf
        edges = set()
        for simplex in tri.simplices:
            pts = points[simplex]
            r = AlphaShapeSeg._circumradius(pts)
            if r < max_r:
                for i in range(3):
                    e = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
                    # Boundary edges appear in exactly one triangle
                    if e in edges:
                        edges.discard(e)
                    else:
                        edges.add(e)

        if not edges:
            return cv2.convexHull(points.astype(np.float32))

        # Collect unique boundary point indices
        boundary_idx = set()
        for e in edges:
            boundary_idx.update(e)
        boundary_pts = points[list(boundary_idx)]

        if len(boundary_pts) < 3:
            return cv2.convexHull(points.astype(np.float32))

        # Order points angularly around centroid
        centroid = boundary_pts.mean(axis=0)
        angles = np.arctan2(boundary_pts[:, 1] - centroid[1],
                            boundary_pts[:, 0] - centroid[0])
        order = np.argsort(angles)
        ordered = boundary_pts[order].astype(np.int32).reshape(-1, 1, 2)

        return ordered

    def segment(self, image, gradient):

        n_labels, cc_labels, img_area, min_area = \
            _interior_connected_components(gradient)

        if n_labels <= 1:
            return np.zeros(gradient.shape, dtype=np.int32)

        # Adaptive alpha: based on typical object scale
        diag = np.sqrt(gradient.shape[0]**2 + gradient.shape[1]**2)
        alpha = 4.0 / diag  # smaller alpha → tighter fit

        hulls = []
        for lbl in range(1, n_labels):
            mask = (cc_labels == lbl).astype(np.uint8)
            px_count = int(mask.sum())
            if px_count < min_area or px_count > 0.5 * img_area:
                continue

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            # Subsample boundary points for speed (max ~800 points)
            all_pts = np.vstack(contours).reshape(-1, 2)
            if len(all_pts) > 800:
                idx = np.linspace(0, len(all_pts) - 1, 800, dtype=int)
                all_pts = all_pts[idx]

            polygon = self._alpha_shape_polygon(all_pts, alpha)
            if polygon is None:
                continue

            area = cv2.contourArea(polygon)
            if area < min_area:
                continue

            hulls.append((area, polygon))

        if not hulls:
            return np.zeros(gradient.shape, dtype=np.int32)

        hulls.sort(key=lambda t: t[0], reverse=True)

        labels = np.zeros(gradient.shape, dtype=np.int32)
        for idx, (_, hull) in enumerate(hulls, start=1):
            cv2.drawContours(labels, [hull], -1, int(idx),
                             thickness=cv2.FILLED)
        return labels


# =====================================================================
# 4. Distance Transform Partitioning
#    Seeds from distance-transform local maxima → watershed partitions.
# =====================================================================

@register_segment("dist_transform")
class DistTransformSeg(SegmentationStrategy):
    """
    Use the distance transform of the interior mask to find local
    maxima (one per object core), then watershed-partition from those
    seeds using the gradient as the flooding landscape.
    """

    def segment(self, image, gradient):
        from scipy.ndimage import maximum_filter, label as scipy_label

        interior = _gradient_to_interior_mask(gradient)
        dist = cv2.distanceTransform(interior, cv2.DIST_L2, 5)

        if dist.max() == 0:
            return np.zeros(gradient.shape, dtype=np.int32)

        # Local maxima detection: a pixel is a peak if it equals the
        # max in its neighbourhood and is above a minimum distance.
        footprint_size = max(5, int(0.02 * min(gradient.shape)))
        if footprint_size % 2 == 0:
            footprint_size += 1
        local_max = maximum_filter(dist, size=footprint_size)

        # Adaptive threshold: use a fraction of the max, with a fallback
        dt_thresh = max(0.1 * dist.max(), 2.0)
        peaks = ((dist == local_max) & (dist > dt_thresh))
        peaks = peaks.astype(np.uint8) * 255

        # Dilate peaks to form solid seed blobs, then clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        peaks = cv2.dilate(peaks, kernel, iterations=2)
        peaks = cv2.morphologyEx(peaks, cv2.MORPH_OPEN, kernel)

        n_seeds, markers = cv2.connectedComponents(peaks)
        if n_seeds <= 1:
            return np.zeros(gradient.shape, dtype=np.int32)

        # Build watershed markers: seeds keep their label,
        # non-interior = background (label n_seeds), unlabeled interior = 0
        markers = markers.astype(np.int32)
        markers = markers + 1                    # shift: bg seed pixels → 1
        markers[interior == 0] = 0               # exterior stays at 0 too

        # Re-mark the non-seed interior as 0 (unknown, to be flooded)
        # Label 1 is the shifted "non-seed" background — mark it unknown
        markers[markers == 1] = 0

        if np.all(markers == 0):
            return np.zeros(gradient.shape, dtype=np.int32)

        grad_bgr = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(grad_bgr, markers)
        markers[markers <= 0] = 0

        # Remove small / background regions
        img_area = gradient.shape[0] * gradient.shape[1]
        return _filter_labels_by_area(markers, markers.max() + 1,
                                      img_area, img_area * 0.003)


# =====================================================================
# 5. Seeded Region Growing
#    Extract seeds from distance peaks, grow by intensity similarity.
# =====================================================================

@register_segment("region_growing")
class SeededRegionGrowingSeg(SegmentationStrategy):
    """
    1. Extract seeds from distance-transform peaks on interior mask.
    2. Iteratively grow each seed by absorbing neighbouring pixels
       whose intensity is within a tolerance of the region mean.
    3. Uses a priority queue (BFS) for deterministic, stable growth.

    Fully vectorized per-iteration using NumPy for parallelism.
    """

    def segment(self, image, gradient):

        interior = _gradient_to_interior_mask(gradient)
        dist = cv2.distanceTransform(interior, cv2.DIST_L2, 5)

        # Seeds: threshold the distance transform
        _, seeds = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
        seeds = np.uint8(seeds)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, kernel)

        n_seeds, seed_labels = cv2.connectedComponents(seeds)
        if n_seeds <= 1:
            return np.zeros(gradient.shape, dtype=np.int32)

        # Use grayscale image for intensity comparison
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        gray = gray.astype(np.float32)

        labels = seed_labels.astype(np.int32).copy()
        h, w = labels.shape

        # Pre-compute region means
        region_sums = np.zeros(n_seeds, dtype=np.float64)
        region_counts = np.zeros(n_seeds, dtype=np.float64)
        for lbl in range(1, n_seeds):
            mask = labels == lbl
            region_sums[lbl] = gray[mask].sum()
            region_counts[lbl] = mask.sum()

        tolerance = 25.0  # intensity tolerance for growth
        max_iterations = 150

        for _ in range(max_iterations):
            # Dilate current labels by 1 pixel
            labels_u16 = labels.astype(np.uint16)
            dilated = cv2.dilate(labels_u16, kernel).astype(np.int32)

            # Candidate pixels: unlabeled, interior, and adjacent to a region
            candidates = (labels == 0) & (dilated > 0) & (interior > 0)
            if not np.any(candidates):
                break

            cand_y, cand_x = np.where(candidates)
            cand_labels = dilated[cand_y, cand_x]
            cand_intensity = gray[cand_y, cand_x]

            # Check intensity tolerance against region mean
            means = np.where(
                region_counts[cand_labels] > 0,
                region_sums[cand_labels] / region_counts[cand_labels],
                0.0
            )
            accept = np.abs(cand_intensity - means) <= tolerance

            if not np.any(accept):
                break

            # Apply accepted pixels
            acc_y = cand_y[accept]
            acc_x = cand_x[accept]
            acc_lbl = cand_labels[accept]
            acc_int = cand_intensity[accept]

            labels[acc_y, acc_x] = acc_lbl

            # Update region stats
            for lbl in np.unique(acc_lbl):
                mask = acc_lbl == lbl
                region_sums[lbl] += acc_int[mask].sum()
                region_counts[lbl] += mask.sum()

        img_area = h * w
        return _filter_labels_by_area(labels, n_seeds, img_area,
                                      img_area * 0.003)


# =====================================================================
# 6. Random Walker Segmentation
#    Probabilistic label diffusion from seeds using scikit-image.
# =====================================================================

@register_segment("random_walker")
class RandomWalkerSeg(SegmentationStrategy):
    """
    Extract seeds from distance-transform peaks, then use scikit-image's
    random_walker to probabilistically diffuse labels across the image.
    Produces very clean boundaries under noise.
    """

    def segment(self, image, gradient):
        from skimage.segmentation import random_walker as rw

        interior = _gradient_to_interior_mask(gradient)
        dist = cv2.distanceTransform(interior, cv2.DIST_L2, 5)

        if dist.max() == 0:
            return np.zeros(gradient.shape, dtype=np.int32)

        # Seeds from distance peaks
        _, seeds = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
        seeds = np.uint8(seeds)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, kernel)

        n_seeds, seed_labels = cv2.connectedComponents(seeds)
        if n_seeds <= 1:
            return np.zeros(gradient.shape, dtype=np.int32)

        # Prepare marker array: 0 = unlabeled, >0 = seed label
        # skimage random_walker segments areas where markers == 0
        markers = seed_labels.astype(np.int32).copy()

        # Ensure there actually are unlabeled pixels
        if not np.any(markers == 0):
            return seed_labels.astype(np.int32)

        # Use grayscale for the random walker
        if len(image.shape) == 3:
            data = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            data = image.copy()

        data = data.astype(np.float64) / 255.0

        try:
            result = rw(data, markers, beta=130, mode='bf',
                        return_full_prob=False)
            result = result.astype(np.int32)
        except Exception:
            # Fall back to seed labels if random walker fails
            result = seed_labels.astype(np.int32)

        img_area = gradient.shape[0] * gradient.shape[1]
        return _filter_labels_by_area(result, result.max() + 1,
                                      img_area, img_area * 0.003)


# =====================================================================
# 7. Split-and-Merge (Quadtree Segmentation)
#    Recursively split into quads, merge homogeneous neighbours.
# =====================================================================

@register_segment("split_merge")
class SplitMergeSeg(SegmentationStrategy):
    """
    Quadtree-based split-and-merge:
        1. Recursively split image quadrants until each is homogeneous
           (std < threshold) or smaller than min_size.
        2. Merge adjacent regions with similar mean intensity.
        3. Filter small/large regions.
    """

    def segment(self, image, gradient):

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        h, w = gray.shape
        img_area = h * w
        min_block = max(8, min(h, w) // 40)
        std_threshold = 18.0

        # ── Split phase: quadtree decomposition ──────────────────────
        labels = np.zeros((h, w), dtype=np.int32)
        current_label = 0
        block_info = {}  # label → (mean_intensity, pixel_count)

        stack = [(0, 0, h, w)]
        while stack:
            r, c, bh, bw = stack.pop()

            block = gray[r:r + bh, c:c + bw]
            if bh <= min_block or bw <= min_block or np.std(block) < std_threshold:
                current_label += 1
                labels[r:r + bh, c:c + bw] = current_label
                block_info[current_label] = (float(np.mean(block)),
                                              int(bh * bw))
            else:
                mh, mw = bh // 2, bw // 2
                stack.append((r,      c,      mh,      mw))
                stack.append((r,      c + mw, mh,      bw - mw))
                stack.append((r + mh, c,      bh - mh, mw))
                stack.append((r + mh, c + mw, bh - mh, bw - mw))

        if current_label == 0:
            return np.zeros((h, w), dtype=np.int32)

        # ── Merge phase: combine adjacent regions with similar mean ──
        merge_threshold = 20.0
        # Build a merge-find (union-find) structure
        parent = list(range(current_label + 1))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Check 4-connected neighbours along label boundaries
        # Vectorized: compare each pixel with its right and bottom neighbour
        # For large images, subsample boundary pixels for speed
        if h > 1:
            diff_v = labels[:-1, :] != labels[1:, :]
            ry, rx = np.where(diff_v)
            if len(ry) > 20000:
                idx = np.random.choice(len(ry), 20000, replace=False)
                ry, rx = ry[idx], rx[idx]
            la_v = labels[ry, rx]
            lb_v = labels[ry + 1, rx]
            for la, lb in zip(la_v.tolist(), lb_v.tolist()):
                if la > 0 and lb > 0:
                    ra, rb = find(la), find(lb)
                    if ra != rb:
                        ma = block_info.get(ra, block_info.get(la, (0, 0)))[0]
                        mb = block_info.get(rb, block_info.get(lb, (0, 0)))[0]
                        if abs(ma - mb) < merge_threshold:
                            union(ra, rb)

        if w > 1:
            diff_h = labels[:, :-1] != labels[:, 1:]
            ry, rx = np.where(diff_h)
            if len(ry) > 20000:
                idx = np.random.choice(len(ry), 20000, replace=False)
                ry, rx = ry[idx], rx[idx]
            la_h = labels[ry, rx]
            lb_h = labels[ry, rx + 1]
            for la, lb in zip(la_h.tolist(), lb_h.tolist()):
                if la > 0 and lb > 0:
                    ra, rb = find(la), find(lb)
                    if ra != rb:
                        ma = block_info.get(ra, block_info.get(la, (0, 0)))[0]
                        mb = block_info.get(rb, block_info.get(lb, (0, 0)))[0]
                        if abs(ma - mb) < merge_threshold:
                            union(ra, rb)

        # Remap labels to roots
        for lbl in range(1, current_label + 1):
            root = find(lbl)
            if root != lbl:
                labels[labels == lbl] = root

        # Compact labels to sequential 1..N
        unique_lbls = np.unique(labels)
        unique_lbls = unique_lbls[unique_lbls > 0]
        remap = np.zeros(labels.max() + 1, dtype=np.int32)
        for new_id, old_id in enumerate(unique_lbls, start=1):
            remap[old_id] = new_id
        labels = remap[labels]

        return _filter_labels_by_area(labels, int(labels.max()) + 1,
                                      img_area, img_area * 0.003)


# =====================================================================
# 8. MSER (Maximally Stable Extremal Regions)
#    OpenCV's built-in detector for stable intensity regions.
# =====================================================================

@register_segment("mser")
class MSERSeg(SegmentationStrategy):
    """
    Detect maximally stable extremal regions using OpenCV.
    Excellent for packaged products with distinct labels/logos.
    Each detected stable region becomes a labelled segment.
    """

    def segment(self, image, gradient):

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape
        img_area = h * w
        min_area = int(img_area * 0.003)

        mser = cv2.MSER_create(
            5,                          # delta
            min_area,                   # min_area
            int(img_area * 0.5),        # max_area
            0.25,                       # max_variation
            0.2,                        # min_diversity
        )

        regions, _ = mser.detectRegions(gray)

        if not regions:
            return np.zeros((h, w), dtype=np.int32)

        # Sort regions by area (largest first), paint smaller on top
        regions = sorted(regions, key=len, reverse=True)

        labels = np.zeros((h, w), dtype=np.int32)
        for idx, region in enumerate(regions, start=1):
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.drawContours(labels, [hull], -1, int(idx),
                             thickness=cv2.FILLED)

        return labels


# =====================================================================
# 9. K-Means (Lab colour + spatial XY features)
#    Colour–position clustering for shelf segmentation.
# =====================================================================

@register_segment("kmeans")
class KMeansSeg(SegmentationStrategy):
    """
    Cluster pixels using K-means on [L, a, b, x, y] features.
    Spatial coordinates are scaled to balance colour and position
    influence. The number of clusters is estimated from the image
    diagonal (more clusters for larger images).
    """

    def segment(self, image, gradient):

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        h, w = image.shape[:2]
        img_area = h * w

        # Convert to Lab colour space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Build feature matrix [L, a, b, x, y]
        yy, xx = np.mgrid[0:h, 0:w]
        spatial_scale = 0.5  # balance colour vs position
        features = np.zeros((img_area, 5), dtype=np.float32)
        features[:, 0] = lab[:, :, 0].ravel()
        features[:, 1] = lab[:, :, 1].ravel()
        features[:, 2] = lab[:, :, 2].ravel()
        features[:, 3] = (xx.ravel().astype(np.float32) / w * 255
                          * spatial_scale)
        features[:, 4] = (yy.ravel().astype(np.float32) / h * 255
                          * spatial_scale)

        # Adaptive K: more clusters for larger / more complex images
        k = max(4, min(30, int(np.sqrt(img_area) / 40)))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    60, 1.0)
        _, labels_flat, _ = cv2.kmeans(
            features, k, None, criteria, attempts=5,
            flags=cv2.KMEANS_PP_CENTERS
        )

        labels = labels_flat.reshape(h, w).astype(np.int32) + 1  # 1-based

        return _filter_labels_by_area(labels, k + 1, img_area,
                                      img_area * 0.003)
    