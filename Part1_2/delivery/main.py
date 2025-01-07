import argparse
import logging
import os
from pathlib import Path
import cv2
import numpy as np
import scipy.io as sio

from ImageNode import ImageNode
from plot import create_canvas_with_images
from save_results import save_homographies_and_yolo
from load_data import process_dir_pairs, load_reference_node
from thresholds import (
    max_descriptor_distance_threshold_numpy,
    compute_min_inliers_threshold,
)
from homography import (
    find_direct_homographies,
    find_remaining_homographies,
)


def setup_logging():
    """Configure the logging settings."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def parse_arguments():
    """Extract the argument parser into its own function."""
    parser = argparse.ArgumentParser(description="Process image sequences.")
    parser.add_argument(
        "ref_dir", type=str, help="Path to the reference image directory"
    )
    parser.add_argument(
        "dirs",
        nargs="+",
        help="Input/output directory pairs: input1 output1 input2 output2 ...",
    )

    args = parser.parse_args()
    ref_dir = Path(args.ref_dir)

    # Ensure the directories come in pairs
    if len(args.dirs) % 2 != 0:
        logging.error("The number of input/output directories must be even.")
        raise ValueError("Invalid input/output directory pairs")

    input_output_pairs = []
    for i in range(0, len(args.dirs), 2):
        inp = Path(args.dirs[i])
        outp = Path(args.dirs[i + 1])
        input_output_pairs.append((inp, outp))

    return ref_dir, input_output_pairs


def main():
    setup_logging()

    # 1) Parse Arguments
    try:
        ref_dir, input_output_pairs = parse_arguments()
    except ValueError as e:
        logging.info("Error parsing arguments.")
        logging.info("Please ensure the input/output directories are valid.")
        logging.info(
            "Usage Example:\n"
            "python main.py ref_dir input1_dir output1_dir input2_dir output2_dir"
        )
        return

    # 2) Load the reference node
    ref_node = load_reference_node(ref_dir)

    # 3) Load all image nodes (from multiple directory pairs)
    image_nodes = process_dir_pairs(input_output_pairs)
    for i, node in enumerate(image_nodes):
        node.image_num = i

    # 4) Compute thresholds
    max_desc_dist = max_descriptor_distance_threshold_numpy(image_nodes, percentile=90)
    logging.info(f"Max descriptor distance (p={90}): {round(max_desc_dist, 4)}")

    min_inliers_threshold = compute_min_inliers_threshold(image_nodes, max_desc_dist)
    logging.info(f"Minimum number of inliers: {min_inliers_threshold}")

    # 5) Find direct homographies
    valid_homographies = find_direct_homographies(
        image_nodes, ref_node, min_inliers_threshold, max_desc_dist
    )

    # if not valid_homographies[1], means no valid homographies
    if not valid_homographies[1]:
        raise ValueError("No valid direct homographies found.")

    # 6) Find remaining homographies if needed
    find_remaining_homographies(
        image_nodes, valid_homographies, min_inliers_threshold, max_desc_dist
    )

    # 7) Transform YOLO
    logging.info("Transforming YOLO detections...")
    for node in image_nodes:
        node.transform_yolo_detections()

    # 8) (Optional) Print footprints or debugging info
    for node in image_nodes:
        logging.info(f"Node {node.idx} -> footprint: {node.footprint}")

    # 9) Save results (homographies + transformed YOLO data)
    save_homographies_and_yolo(image_nodes)

    # 10) Save each warped image using the ImageNode method
    for node in image_nodes:
        warped_image = node.get_warped_image_with_yolo()
        if warped_image is not None and node.output_path is not None:
            out_path = node.output_path / f"warped_{node.idx}.jpg"
            cv2.imwrite(str(out_path), warped_image)

    # 11) Save the final canvas
    final_canvas = create_canvas_with_images(ref_node.image, image_nodes)
    if input_output_pairs:
        canvas_out = input_output_pairs[0][1] / "final_canvas.jpg"
        cv2.imwrite(str(canvas_out), final_canvas)

    logging.info("Processing complete! Results saved.")


if __name__ == "__main__":
    main()
