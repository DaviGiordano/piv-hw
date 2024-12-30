import cv2
import numpy as np
import open3d as o3d


def draw_image_keypoints(img, keypoints, color=(1, 0, 0), radius=3, thickness=-1):
    """
    Overlays keypoints on a copy of the given image and returns the result.

    Args:
        img (np.ndarray): Input image (grayscale or BGR).
        keypoints (list[cv2.KeyPoint]): List of keypoints.
        color (tuple): BGR color for drawing.
        radius (int): Radius of the keypoint circle.
        thickness (int): Thickness of the circle outline, or -1 for a filled circle.

    Returns:
        np.ndarray: Image with keypoints drawn.
    """

    # Convert grayscale to BGR if needed
    if len(img.shape) == 2:
        display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        display_img = img.copy()

    for kp in keypoints:
        x, y = kp
        cv2.circle(display_img, (int(x), int(y)), radius, color, thickness)

    return display_img


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw_matches_points(img1, img2, src_points, dst_points, max_points=None):
    """
    Draw lines between corresponding points from two images, placed side by side.

    Args:
        img1 (np.ndarray): First image (Grayscale or BGR).
        img2 (np.ndarray): Second image (Grayscale or BGR).
        src_points (np.ndarray): Nx2 array of points in the first image.
        dst_points (np.ndarray): Nx2 array of points in the second image.
        max_points (int or None): If int, maximum number of points to draw.
                                  If None, draw all points.
    Returns:
        np.ndarray: Side-by-side visualization of the two images with lines
                    connecting matching points.
    """
    # Convert to BGR if needed
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        # Make a copy in case of in-place modifications
        img1 = img1.copy()

    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2 = img2.copy()

    # Ensure both are uint8 in [0..255]
    if img1.dtype != np.uint8:
        img1 = (
            (img1 * 255).astype(np.uint8)
            if img1.max() <= 1.0
            else img1.astype(np.uint8)
        )
    if img2.dtype != np.uint8:
        img2 = (
            (img2 * 255).astype(np.uint8)
            if img2.max() <= 1.0
            else img2.astype(np.uint8)
        )

    # Create a canvas for side-by-side
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place img1 on the left, img2 on the right
    canvas[:h1, :w1] = img1
    canvas[:h2, w1 : w1 + w2] = img2

    # Decide how many points to draw
    total_points = min(len(src_points), len(dst_points))
    num_points = total_points if max_points is None else min(total_points, max_points)

    for i in range(num_points):
        x1, y1 = src_points[i]
        x2, y2 = dst_points[i]
        x2_offset = x2 + w1  # shift the second image's x-coordinate

        # Pick a random color in 0..255
        color = tuple(np.random.randint(0, 256, size=3).tolist())

        # Draw circles and a line
        cv2.circle(canvas, (int(x1), int(y1)), 3, color, -1)
        cv2.circle(canvas, (int(x2_offset), int(y2)), 3, color, -1)
        cv2.line(canvas, (int(x1), int(y1)), (int(x2_offset), int(y2)), color, 1)

    return canvas


def register_point_cloud(points, cloud_colors) -> o3d.geometry.PointCloud:

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(cloud_colors)
    return pcd


def write_point_cloud(pcd, output_fpath):
    o3d.io.write_point_cloud(output_fpath, pcd, write_ascii=True)


def plot_point_cloud_with_keypoints(
    points,
    keypoints_3d,
    point_color=(0.5, 0.5, 0.5),
    keypoint_color=(1.0, 0.0, 0.0),
    output_prefix="./output",
):
    """
    Visualizes a point cloud in grayscale/black-and-white, with keypoints in red.
    Uses Open3D for rendering, but writes two separate files (one for cloud, one for keypoints).
    """

    # Main cloud
    pcd = register_point_cloud(points, np.tile(point_color, (points.shape[0], 1)))
    kp_pcd = register_point_cloud(
        points, np.tile(keypoints_3d, (keypoints_3d.shape[0], 1))
    )

    # Writes two separate files
    write_point_cloud(pcd, f"{output_prefix}_cloud.ply")
    write_point_cloud(kp_pcd, f"{output_prefix}_keypoints.ply")
