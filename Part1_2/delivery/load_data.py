import cv2
from scipy.io import loadmat
from pathlib import Path
from PIL import Image
import numpy as np
from ImageNode import ImageNode

# SCALE_FACTOR = 0.5


# def resize_keypoint_and_image(keypoint, image, scale_factor):

#     resized_keypoint = keypoint * scale_factor
#     resized_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

#     return resized_keypoint, resized_image


def parse_yolo(yolo_fpath):
    """
    Loads YOLO detection data from the specified .mat file.

    Args:
        yolo_fpath (Path): File path to the YOLO .mat file.

    Returns:
        dict: YOLO detection data loaded from the file.
    """
    if yolo_fpath:
        yolo_data = loadmat(yolo_fpath)
        yolo_dict = {
            "id": yolo_data["id"],
            "class": yolo_data["class"],
            "xyxy": yolo_data["xyxy"],
        }
        return yolo_dict
    else:
        return None


def parse_kps(kps_fpath):
    """
    Loads keypoints and descriptors from a .mat file.

    Args:
        kps_fpath (Path): File path to the keypoints .mat file.

    Returns:
        tuple: A tuple of (keypoints, descriptors).
    """
    keypoints = loadmat(kps_fpath)["kp"]
    descriptors = loadmat(kps_fpath)["desc"]
    return keypoints, descriptors


def get_sorted_file_paths(input_path, pattern):
    """
    Retrieves file paths matching a pattern, sorted by their numeric ID in the filename.

    Args:
        input_path (Path): Directory containing the files.
        pattern (str): Glob pattern (e.g., '*.jpg').

    Returns:
        list: Sorted list of file paths.
    """
    return sorted(input_path.glob(pattern), key=lambda x: int(x.stem.split("_")[1]))


def build_file_path_dict(file_paths):
    """
    Builds a dictionary mapping numeric IDs to file paths.

    Args:
        file_paths (list): List of file paths to include.

    Returns:
        dict: { numeric_id: file_path }
    """
    return {int(fpath.stem.split("_")[1]): fpath for fpath in file_paths}


def process_image_node(
    img_fpath, kps_fpaths_dict, yolo_fpaths_dict, input_path, output_path
):
    """
    Creates an ImageNode for a given image, loading keypoints, descriptors, and YOLO data.

    Args:
        img_fpath (Path): Path to the image file.
        kps_fpaths_dict (dict): Dictionary of keypoint file paths.
        yolo_fpaths_dict (dict): Dictionary of YOLO file paths.
        input_path (Path): The parent input directory.
        output_path (Path): The parent output directory.

    Returns:
        ImageNode: An ImageNode instance with loaded data.
    """
    img_num = int(img_fpath.stem.split("_")[1])
    kps_fpath = kps_fpaths_dict.get(img_num, None)
    yolo_fpath = yolo_fpaths_dict.get(img_num, None)
    keypoints, descriptors = parse_kps(kps_fpath)
    image = np.array(Image.open(img_fpath))
    # keypoints, image = resize_keypoint_and_image(keypoints, image, SCALE_FACTOR)

    return ImageNode(
        image=image,
        keypoints=keypoints,
        descriptors=descriptors,
        idx=f"{input_path.parts[-1]}_{img_num}",
        input_fpath=img_fpath,
        output_path=output_path,
        yolo_detections=parse_yolo(yolo_fpath),
    )


def load_dir(input_path, output_path):
    """
    Loads all relevant image files in a directory, creating ImageNode objects.

    Args:
        input_path (Path): Directory containing the images and .mat files.
        output_path (Path): Directory where outputs might be stored or saved.

    Returns:
        list: A list of ImageNode objects for all processed images.
    """
    imgs_fpaths = get_sorted_file_paths(input_path, "*.jpg")
    kps_fpaths = get_sorted_file_paths(input_path, "kp*.mat")
    yolo_fpaths = get_sorted_file_paths(input_path, "yolo*.mat")

    kps_fpaths_dict = build_file_path_dict(kps_fpaths)
    yolo_fpaths_dict = build_file_path_dict(yolo_fpaths)

    image_nodes_curr_dir = [
        process_image_node(
            img_fpath, kps_fpaths_dict, yolo_fpaths_dict, input_path, output_path
        )
        for img_fpath in imgs_fpaths
    ]

    return image_nodes_curr_dir


def process_dir_pairs(dir_pairs):
    """
    Processes multiple (input, output) directory pairs, aggregating all ImageNodes.

    Args:
        dir_pairs (list of tuple): List of (input_dir, output_dir) pairs.

    Returns:
        list: A combined list of ImageNode objects from all pairs.
    """
    all_image_nodes = []

    for input_dir, output_dir in dir_pairs:
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        image_nodes = load_dir(input_path, output_path)
        all_image_nodes.extend(image_nodes)

    return all_image_nodes


def load_reference_node(ref_dir):
    """
    Load the reference image and keypoints/descriptors from the specified directory.

    Args:
        ref_dir (Path): Directory containing the reference image and keypoints/descriptors.

    Returns:
        ImageNode: An ImageNode instance representing the reference image.
    """
    ref_img_fpath = ref_dir / "img_ref.jpg"
    ref_kps_fpath = ref_dir / "kp_ref.mat"

    ref_img = np.array(Image.open(ref_img_fpath))
    ref_data = loadmat(ref_kps_fpath)
    ref_keypoints = ref_data["kp"]
    ref_descriptors = ref_data["desc"]

    ref_node = ImageNode(
        image=ref_img,
        keypoints=ref_keypoints,
        descriptors=ref_descriptors,
        idx="ref",
        input_fpath=ref_img_fpath,
        output_path=None,
        yolo_detections=None,
    )
    return ref_node
