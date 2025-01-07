import argparse
import logging
import os
from pathlib import Path
from scipy.io import loadmat
from PIL import Image
import numpy as np
from tqdm import tqdm
from itertools import chain

DISTANCE_THRESHOLD_PERCENTILE = 90
RANSAC_THRESHOLD = 5
SCALE_FACTOR = 0.5


class ImageNode:

    def __init__(self, image, keypoints, descriptors, idx):
        self.image = image
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.idx = idx
        self.homography = None
        self.footprint = [idx]

    def __str__(self):
        return f"ImageNode(idx={self.idx}, keypoints_shape={self.keypoints.shape}, descriptors_shape={self.descriptors.shape}, footprint={self.footprint})"


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")


def create_output_directory(output_path):
    output_path.mkdir(parents=True, exist_ok=True)


def load_reference_data(ref_dir):
    """Load the reference image and keypoints/descriptors."""
    logging.info(f"Loading reference data from {ref_dir}")

    ref_img_fpath = ref_dir / "img_ref.jpg"
    ref_kps_fpath = ref_dir / "kp_ref.mat"

    ref_img = np.array(Image.open(ref_img_fpath))

    ref_data = loadmat(ref_kps_fpath)
    ref_keypoints = ref_data['kp']
    ref_descriptors = ref_data['desc']

    logging.info(f"Loaded reference image with shape {ref_img.shape}")
    logging.info(f"Loaded {len(ref_keypoints)} reference keypoints")

    return ref_img, ref_keypoints, ref_descriptors


def load_input_data(input_dir):
    """Load images, keypoints/descriptors, and YOLO detections from the input directory."""
    logging.info(f"Loading input data from {input_dir}")

    imgs_fpaths = sorted(input_dir.glob("*.jpg"),
                         key=lambda x: int(x.stem.split('_')[1]))
    images = [np.array(Image.open(fpath)) for fpath in imgs_fpaths]

    kps_fpaths = sorted(input_dir.glob("kp*.mat"),
                        key=lambda x: int(x.stem.split('_')[1]))
    keypoints = [loadmat(fpath)['kp'] for fpath in kps_fpaths]
    descriptors = [loadmat(fpath)['desc'] for fpath in kps_fpaths]

    yolo_fpaths = sorted(input_dir.glob("yolo*.mat"),
                         key=lambda x: int(x.stem.split('_')[1]))
    yolo_data = []
    for fpath in yolo_fpaths:
        data = loadmat(fpath)
        yolo_data.append({
            'xyxy': data.get('xyxy', np.array([])),
            'id': data.get('id', np.array([])),
            'class': data.get('class', np.array([]))
        })

    logging.info(f"Loaded {len(images)} images from {input_dir}")
    logging.info(f"Loaded {len(keypoints)} keypoint files from {input_dir}")
    logging.info(
        f"Loaded {len(yolo_data)} YOLO detection files from {input_dir}")

    return images, keypoints, descriptors, yolo_data


def process_directories(ref_dir, input_output_pairs):
    """Process the directories and return a structured dictionary."""
    # Load reference data
    ref_img, ref_keypoints, ref_descriptors = load_reference_data(ref_dir)

    # Prepare the data dictionary
    data = {
        "reference": {
            "image": ref_img,
            "keypoints": ref_keypoints,
            "descriptors": ref_descriptors
        },
        "inputs": []
    }

    for input_dir, output_dir in input_output_pairs:
        logging.info(f"Processing input directory: {input_dir}")
        logging.info(f"Storing results in output directory: {output_dir}")

        if not input_dir.exists():
            logging.error(f"Input directory '{input_dir}' does not exist.")
            continue

        # Ensure output directory exists
        create_output_directory(output_dir)

        # Load input data
        images, keypoints, descriptors, yolo_data = load_input_data(input_dir)

        # Add to the data dictionary
        data["inputs"].append({
            "images": images,
            "keypoints": keypoints,
            "descriptors": descriptors,
            "yolo_data": yolo_data,
            "output_path": output_dir
        })

        logging.info(f"Finished processing {input_dir}")

    return data


def parse_arguments():
    """Extract the argument parser into its own function."""
    parser = argparse.ArgumentParser(description="Process image sequences.")
    parser.add_argument("ref_dir",
                        type=str,
                        help="Path to the reference image directory")
    parser.add_argument("dirs", nargs="+", help="Input/output directory pairs")
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir)

    if len(args.dirs) % 2 != 0:
        logging.error("The number of input/output directories must be even.")
        raise ValueError("Invalid input/output directory pairs")

    input_output_pairs = [(Path(args.dirs[i]), Path(args.dirs[i + 1]))
                          for i in range(0, len(args.dirs), 2)]

    return ref_dir, input_output_pairs


############################################
# THRESHOLD FUNCTIONS
############################################


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
        src_h = np.hstack([src_pts, np.ones(
            (num_points, 1))])  # Homogeneous coordinates
        dst_proj = src_h @ H.T
        dst_proj /= dst_proj[:, 2:3]  # Normalize to get (x, y)

        # Efficient squared distance calculation with safeguard
        distances = np.sqrt(
            np.maximum(
                0,
                np.sum(dst_pts**2, axis=1) +
                np.sum(dst_proj[:, :2]**2, axis=1) -
                2 * np.sum(dst_pts * dst_proj[:, :2], axis=1)))

        # Compute inliers based on the threshold
        inlier_mask = distances < ransac_threshold
        num_inliers = np.sum(inlier_mask)

        # Update the best model
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H
            best_inlier_mask = inlier_mask

    return best_H, best_inlier_mask


def find_matches_numpy(src_node, dst_node, max_desc_dist=None):
    """
    Find matches between two ImageNode objects using numpy with cross-check.

    Args:
        src_node (ImageNode): Source image node.
        dst_node (ImageNode): Destination image node.
        max_desc_dist (float, optional): Maximum descriptor distance for valid matches.

    Returns:
        src_pts (np.ndarray): Matched keypoints from the source node.
        dst_pts (np.ndarray): Matched keypoints from the destination node.
    """
    # Extract descriptors
    src_descriptors = src_node.descriptors
    dst_descriptors = dst_node.descriptors

    distances = np.sqrt(
        np.maximum(
            0,
            np.sum(src_descriptors[:, None, :]**2, axis=2) +
            np.sum(dst_descriptors[None, :, :]**2, axis=2) -
            2 * np.dot(src_descriptors, dst_descriptors.T)))

    # Find the nearest neighbor in both directions
    src_to_dst = np.argmin(distances, axis=1)
    dst_to_src = np.argmin(distances, axis=0)

    # Reciprocal matching (cross-check logic)
    reciprocal_matches = [(src_idx, dst_idx)
                          for src_idx, dst_idx in enumerate(src_to_dst)
                          if dst_to_src[dst_idx] == src_idx]

    # Apply distance threshold if provided
    if max_desc_dist is not None:
        reciprocal_matches = [(src_idx, dst_idx)
                              for src_idx, dst_idx in reciprocal_matches
                              if distances[src_idx, dst_idx] <= max_desc_dist]

    # Extract matched keypoints
    src_pts = np.array(
        [src_node.keypoints[src_idx] for src_idx, _ in reciprocal_matches],
        dtype=np.float32)
    dst_pts = np.array(
        [dst_node.keypoints[dst_idx] for _, dst_idx in reciprocal_matches],
        dtype=np.float32)

    return src_pts, dst_pts


def max_descriptor_distance_threshold_numpy(image_nodes,
                                            distance_threshold_percentile=100):
    all_thresholds = []

    for i in tqdm(range(len(image_nodes) - 1)):
        src_node = image_nodes[i]
        dst_node = image_nodes[i + 1]

        distances = np.sqrt(
            np.maximum(
                0,
                np.sum(src_node.descriptors[:, None, :]**2, axis=2) +
                np.sum(dst_node.descriptors[None, :, :]**2, axis=2) -
                2 * np.dot(src_node.descriptors, dst_node.descriptors.T)))

        # Cross-check logic
        src_to_dst = np.argmin(distances, axis=1)
        dst_to_src = np.argmin(distances, axis=0)
        reciprocal_matches = [(src_idx, dst_idx)
                              for src_idx, dst_idx in enumerate(src_to_dst)
                              if dst_to_src[dst_idx] == src_idx]

        # Collect distances for reciprocal matches
        match_distances = [
            distances[src_idx, dst_idx]
            for src_idx, dst_idx in reciprocal_matches
        ]

        # Calculate the percentile threshold for current image pair
        if match_distances:
            curr_threshold = np.percentile(match_distances,
                                           distance_threshold_percentile)
            all_thresholds.append(curr_threshold)

    # Return the maximum threshold across all image pairs
    descriptor_distance_threshold = max(
        all_thresholds) if all_thresholds else 0.0
    return descriptor_distance_threshold


def compute_min_inliers_threshold(image_nodes, max_desc_dist):
    inliers_per_pair = []

    for i in tqdm(range(len(image_nodes) - 1)):
        src_pts, dst_pts = find_matches_numpy(image_nodes[i],
                                              image_nodes[i + 1], max_desc_dist)
        H, mask = compute_homography_ransac(src_pts, dst_pts, RANSAC_THRESHOLD)
        inliers = np.sum(mask)
        inliers_per_pair.append(inliers)

    min_inliers = min(inliers_per_pair)
    return min_inliers


############################################
# END THRESHOLD FUNCTIONS
############################################

############################################
# HOMOGRAPHY FUNCTIONS
############################################


def find_valid_homography(src_node, dst_node, min_inliers_threshold,
                          max_desc_dist):
    src_pts, dst_pts = find_matches_numpy(src_node, dst_node, max_desc_dist)
    H, mask = compute_homography_ransac(src_pts, dst_pts, RANSAC_THRESHOLD)
    num_inliers = np.sum(mask)
    is_H_valid = 1 if num_inliers >= min_inliers_threshold else 0
    return H, is_H_valid, num_inliers


def find_direct_homographies(image_nodes, image_node_ref, min_inliers_threshold,
                             max_desc_dist):
    valid_homographies = {1: []}

    for i in tqdm(range(len(image_nodes))):
        H, is_H_valid, num_inliers = find_valid_homography(
            image_nodes[i], image_node_ref, min_inliers_threshold,
            max_desc_dist)

        if is_H_valid:
            image_nodes[i].homography = H
            image_nodes[i].footprint.append('ref')
            valid_homographies[1].append(i)
    return valid_homographies


def find_remaining_homographies(image_nodes, valid_homographies,
                                min_inliers_threshold, max_desc_dist):
    level = 1
    while level in valid_homographies:
        best_idxs = valid_homographies[level]
        for node in tqdm(image_nodes):
            if node.idx in list(chain.from_iterable(
                    valid_homographies.values())):
                continue

            nearest_idx = min(best_idxs, key=lambda x: abs(x - node.idx))
            H, is_H_valid, num_inliers = find_valid_homography(
                node, image_nodes[nearest_idx], min_inliers_threshold,
                max_desc_dist)

            if is_H_valid:
                node.homography = np.dot(image_nodes[nearest_idx].homography, H)
                valid_homographies.setdefault(level + 1, []).append(node.idx)
                node.footprint.extend(image_nodes[nearest_idx].footprint)
        level += 1


############################################
# END HOMOGRAPHY FUNCTIONS
############################################


def create_image_nodes(data):
    image_nodes_list = []
    for input_data in data["inputs"]:
        image_nodes = [
            ImageNode(input_data["images"][j],
                      input_data["keypoints"][j],
                      input_data["descriptors"][j],
                      idx=j) for j in range(len(input_data["images"]))
        ]
        image_nodes_list.append(image_nodes)

    ref_node = ImageNode(data["reference"]["image"],
                         data["reference"]["keypoints"],
                         data["reference"]["descriptors"],
                         idx=-1)
    logging.info(
        f"Created {sum(len(nodes) for nodes in image_nodes_list)} ImageNodes across {len(image_nodes_list)} input directories."
    )
    return image_nodes_list, ref_node


def compute_pipeline(image_nodes: list[ImageNode], ref_node: ImageNode):
    logging.info(f"Created {len(image_nodes)} ImageNodes.")
    descriptor_threshold = max_descriptor_distance_threshold_numpy(
        image_nodes, DISTANCE_THRESHOLD_PERCENTILE)
    logging.info(f"Descriptor distance threshold: {descriptor_threshold}")
    min_inliers = compute_min_inliers_threshold(image_nodes,
                                                descriptor_threshold)
    logging.info(f"Minimum inliers threshold: {min_inliers}")
    valid_homographies = find_direct_homographies(image_nodes, ref_node,
                                                  min_inliers,
                                                  descriptor_threshold)
    find_remaining_homographies(image_nodes, valid_homographies, min_inliers,
                                descriptor_threshold)
    logging.info("Homography estimation complete.")


def main():
    setup_logging()

    try:
        ref_dir, input_output_pairs = parse_arguments()
    except ValueError as e:
        logging.info(
            "Error parsing arguments. Please ensure the input/output directories are valid."
        )
        logging.info("You should call the script like this:")
        logging.info(
            "python main.py <reference_directory> <input_directory> <output_directory> <input_directory> <output_directory>"
        )
        return

    # Process the directories and get the data
    data = process_directories(ref_dir, input_output_pairs)
    image_nodes_list, ref_node = create_image_nodes(data)
    for image_nodes in image_nodes_list:
        compute_pipeline(image_nodes, ref_node)
        # TODO: Get the homographies
        # TODO: Process the YOLO ?

    logging.info("Processing complete")


if __name__ == "__main__":
    main()
