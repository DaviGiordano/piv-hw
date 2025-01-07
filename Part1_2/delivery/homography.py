from itertools import chain
from tqdm import tqdm
import numpy as np

from matches import find_matches_numpy


RANSAC_THRESHOLD = 5


def compute_homography_ransac(src_pts, dst_pts, ransac_threshold):
    """
    Compute homography using numpy with RANSAC.

    Args:
        src_pts (np.ndarray): Source points (Nx2).
        dst_pts (np.ndarray): Destination points (Nx2).
        ransac_threshold (float): RANSAC inlier threshold.

    Returns:
        H (np.ndarray): Normalized homography matrix.
        inlier_mask (np.ndarray): Boolean array of inliers.
    """
    max_inliers = 0
    best_H = None
    best_inlier_mask = None
    num_points = src_pts.shape[0]
    iterations = 1000

    for _ in range(iterations):
        # Randomly sample 4 points
        indices = np.random.choice(num_points, 4, replace=False)
        sampled_src = src_pts[indices]
        sampled_dst = dst_pts[indices]

        # Solve for H using the sampled points
        A = []
        for (x, y), (x_p, y_p) in zip(sampled_src, sampled_dst):
            A.append([-x, -y, -1, 0, 0, 0, x * x_p, y * x_p, x_p])
            A.append([0, 0, 0, -x, -y, -1, x * y_p, y * y_p, y_p])
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)

        # Normalize the homography matrix (ensure H[2, 2] = 1)
        if H[2, 2] != 0:
            H /= H[2, 2]

        # Compute inliers using optimized norm distance calculation
        src_h = np.hstack(
            [src_pts, np.ones((num_points, 1))]
        )  # Homogeneous coordinates
        dst_proj = src_h @ H.T
        dst_proj /= dst_proj[:, 2:3]  # Normalize to get (x, y)

        # Efficient squared distance calculation with safeguard
        distances = np.sqrt(
            np.maximum(
                0,
                np.sum(dst_pts**2, axis=1)
                + np.sum(dst_proj[:, :2] ** 2, axis=1)
                - 2 * np.sum(dst_pts * dst_proj[:, :2], axis=1),
            )
        )

        # Compute inliers based on the threshold
        inlier_mask = distances < ransac_threshold
        num_inliers = np.sum(inlier_mask)

        # Update the best model
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H
            best_inlier_mask = inlier_mask

    return best_H, best_inlier_mask


def find_valid_homography(src_node, dst_node, min_inliers_threshold, max_desc_dist):
    src_pts, dst_pts = find_matches_numpy(src_node, dst_node, max_desc_dist)
    H, mask = compute_homography_ransac(src_pts, dst_pts, RANSAC_THRESHOLD)
    num_inliers = np.sum(mask)
    is_H_valid = 1 if num_inliers >= min_inliers_threshold else 0
    return H, is_H_valid, num_inliers


def find_direct_homographies(
    image_nodes, image_node_ref, min_inliers_threshold, max_desc_dist
):
    valid_homographies = {1: []}

    for i in tqdm(range(len(image_nodes)), desc="Finding direct homographies.."):
        H, is_H_valid, num_inliers = find_valid_homography(
            image_nodes[i], image_node_ref, min_inliers_threshold, max_desc_dist
        )

        if is_H_valid:
            image_nodes[i].homography = H
            image_nodes[i].footprint.append("ref")
            valid_homographies[1].append(image_nodes[i].image_num)

    print("Nodes with direct homography:", valid_homographies)
    return valid_homographies


def find_remaining_homographies(
    image_nodes, valid_homographies, min_inliers_threshold, max_desc_dist
):
    level = 1
    while level in valid_homographies:
        print(f"\n--- Current graph depth: {level} ---")
        print("Nodes at each level:", valid_homographies)

        used_img_nums = list(chain.from_iterable(valid_homographies.values()))

        remaining_nodes = [
            node for node in image_nodes if node.image_num not in used_img_nums
        ]

        print(
            f"Remaining nodes to process: {[node.image_num for node in remaining_nodes]}"
        )

        best_img_nums = valid_homographies[level]

        for node in tqdm(remaining_nodes, desc="Finding remaining homographies.."):
            nearest_img_num = min(best_img_nums, key=lambda x: abs(x - node.image_num))
            H, is_H_valid, num_inliers = find_valid_homography(
                node, image_nodes[nearest_img_num], min_inliers_threshold, max_desc_dist
            )

            if is_H_valid:

                node.homography = np.dot(image_nodes[nearest_img_num].homography, H)
                node.footprint.extend(image_nodes[nearest_img_num].footprint)

                valid_homographies.setdefault(level + 1, []).append(node.image_num)

        level += 1
