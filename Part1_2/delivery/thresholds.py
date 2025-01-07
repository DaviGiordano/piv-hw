import numpy as np
from tqdm import tqdm

from homography import compute_homography_ransac
from matches import find_matches_numpy

RANSAC_THRESHOLD = 5


def max_descriptor_distance_threshold_numpy(
    image_nodes, distance_threshold_percentile=100
):
    all_thresholds = []

    for i in tqdm(
        range(len(image_nodes) - 1), "Calculating max desc distance threshold.."
    ):
        src_node = image_nodes[i]
        dst_node = image_nodes[i + 1]

        distances = np.sqrt(
            np.maximum(
                0,
                np.sum(src_node.descriptors[:, None, :] ** 2, axis=2)
                + np.sum(dst_node.descriptors[None, :, :] ** 2, axis=2)
                - 2 * np.dot(src_node.descriptors, dst_node.descriptors.T),
            )
        )

        # Cross-check logic
        src_to_dst = np.argmin(distances, axis=1)
        dst_to_src = np.argmin(distances, axis=0)
        reciprocal_matches = [
            (src_idx, dst_idx)
            for src_idx, dst_idx in enumerate(src_to_dst)
            if dst_to_src[dst_idx] == src_idx
        ]

        # Collect distances for reciprocal matches
        match_distances = [
            distances[src_idx, dst_idx] for src_idx, dst_idx in reciprocal_matches
        ]

        # Calculate the percentile threshold for current image pair
        if match_distances:
            curr_threshold = np.percentile(
                match_distances, distance_threshold_percentile
            )
            all_thresholds.append(curr_threshold)

    # Return the maximum threshold across all image pairs
    descriptor_distance_threshold = max(all_thresholds) if all_thresholds else 0.0
    return descriptor_distance_threshold


def compute_min_inliers_threshold(image_nodes, max_desc_dist):
    inliers_per_pair = []

    for i in tqdm(range(len(image_nodes) - 1), "Calculating min inliers threshold.."):
        src_pts, dst_pts = find_matches_numpy(
            image_nodes[i], image_nodes[i + 1], max_desc_dist
        )
        H, mask = compute_homography_ransac(src_pts, dst_pts, RANSAC_THRESHOLD)
        inliers = np.sum(mask)
        inliers_per_pair.append(inliers)

    min_inliers = min(inliers_per_pair)
    return min_inliers
