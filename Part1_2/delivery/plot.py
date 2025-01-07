from matplotlib import pyplot as plt
import numpy as np
import cv2

import cv2
import numpy as np
import matplotlib.pyplot as plt

from homography import compute_homography_ransac
from matches import find_matches_numpy

RANSAC_THRESHOLD = 5


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
    # Convert to color if needed
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1 = img1.copy()

    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2 = img2.copy()

    # Create a canvas for side-by-side view
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[0:h1, 0:w1] = img1
    canvas[0:h2, w1 : w1 + w2] = img2

    # Decide how many points to draw
    total_points = min(len(src_points), len(dst_points))
    num_points = total_points if max_points is None else min(total_points, max_points)

    for i in range(num_points):
        x1, y1 = src_points[i]
        x2, y2 = dst_points[i]
        x2_offset = x2 + w1  # shift the second image's x-coord

        color = tuple(np.random.randint(0, 255, size=3).tolist())
        cv2.circle(canvas, (int(x1), int(y1)), 1, color, -1)
        cv2.circle(canvas, (int(x2_offset), int(y2)), 1, color, -1)
        cv2.line(canvas, (int(x1), int(y1)), (int(x2_offset), int(y2)), color, 1)

    return canvas


def visualize_homography_alignment_single(ref_img, node):
    """
    Warp the image to the reference frame using the computed homography
    and plot the warped image alongside the reference without cropping any content.
    """
    if node.homography is not None:
        # Ensure homography is of type float32
        H = node.homography.astype(np.float32)

        # Get the shapes of the reference and current images
        h_ref, w_ref = ref_img.shape[:2]
        h_warp, w_warp = node.image.shape[:2]

        # Define the corners of the reference and current images
        corners_ref = np.array(
            [[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]], dtype=np.float32
        ).reshape(-1, 1, 2)

        corners_warp = np.array(
            [[0, 0], [w_warp, 0], [w_warp, h_warp], [0, h_warp]], dtype=np.float32
        ).reshape(-1, 1, 2)

        # Warp the corners of the current image to the reference frame
        warped_corners = cv2.perspectiveTransform(corners_warp, H)

        # Combine corners to find the overall bounds
        all_corners = np.concatenate((corners_ref, warped_corners), axis=0)

        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        # Compute the translation needed to shift the images
        translation = [-xmin, -ymin]

        # Define the translation matrix with float32 type
        translation_matrix = np.array(
            [[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]],
            dtype=np.float32,
        )

        # Warp the reference image using the translation matrix
        ref_warped = cv2.warpPerspective(
            ref_img, translation_matrix, (xmax - xmin, ymax - ymin)
        )

        # Warp the current image using the combined translation and homography matrices
        combined_homography = translation_matrix @ H  # Matrix multiplication
        warped_img = cv2.warpPerspective(
            node.image, combined_homography, (xmax - xmin, ymax - ymin)
        )

        # Create an overlay by blending the two images
        # Convert images to float for blending
        ref_float = ref_warped.astype(np.float32)
        warped_float = warped_img.astype(np.float32)

        # Normalize images to range [0,1] if they are not already
        if ref_float.max() > 1.0:
            ref_float /= 255.0
        if warped_float.max() > 1.0:
            warped_float /= 255.0

        # Handle different number of channels
        if ref_float.ndim == 2:
            ref_float = cv2.cvtColor(ref_float, cv2.COLOR_GRAY2RGB)
        if warped_float.ndim == 2:
            warped_float = cv2.cvtColor(warped_float, cv2.COLOR_GRAY2RGB)

        # Blend with 50-50 transparency
        overlay = cv2.addWeighted(ref_float, 0.5, warped_float, 0.5, 0)
        overlay = (overlay * 255).astype(np.uint8)

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))
        # Show the reference warped image
        axes[0].imshow(ref_warped, cmap="gray" if ref_warped.ndim == 2 else None)
        axes[0].set_title("Reference Image (Warped to Canvas)")
        axes[0].axis("off")

        # Show the warped current image
        axes[1].imshow(warped_img, cmap="gray" if warped_img.ndim == 2 else None)
        axes[1].set_title(f"Warped Image (idx={node.idx})")
        axes[1].axis("off")

        # Show the overlay
        axes[2].imshow(overlay, cmap="gray" if overlay.ndim == 2 else None)
        axes[2].set_title("Overlay (Ref & Warped)")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()


def create_canvas_with_images(ref_img, image_nodes):
    """
    Create a canvas with all images warped into the reference frame.

    Args:
        ref_img (np.ndarray): The reference image.
        image_nodes (list of ImageNode): List of ImageNode objects, each with an image, keypoints, descriptors, and homography.

    Returns:
        canvas (np.ndarray): The final canvas with all images aligned in the reference frame.
    """
    # Step 1: Determine the overall canvas size by transforming the corners of all images
    h_ref, w_ref = ref_img.shape[:2]

    # Get the corners of the reference image
    corners_ref = np.array(
        [[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]], dtype=np.float32
    ).reshape(-1, 1, 2)

    all_corners = [corners_ref]

    for node in image_nodes:
        if node.homography is not None:
            h_img, w_img = node.image.shape[:2]
            corners_img = np.array(
                [[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]], dtype=np.float32
            ).reshape(-1, 1, 2)

            # Warp the image corners to the reference frame
            warped_corners = cv2.perspectiveTransform(corners_img, node.homography)
            all_corners.append(warped_corners)

    # Combine all the corners to compute the overall canvas size
    all_corners = np.concatenate(all_corners, axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Compute the translation to shift everything into positive canvas space
    translation = [-xmin, -ymin]

    # Step 2: Create the translation matrix to shift all images
    translation_matrix = np.array(
        [[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]], dtype=np.float32
    )

    # Step 3: Warp the reference image into the new canvas
    canvas_size = (xmax - xmin, ymax - ymin)
    canvas = np.zeros(
        (canvas_size[1], canvas_size[0], 3), dtype=np.uint8
    )  # Create an empty canvas

    # Warp the reference image
    ref_warped = cv2.warpPerspective(ref_img, translation_matrix, canvas_size)

    # Add the reference image to the canvas
    mask_ref = (ref_warped > 0).astype(
        np.uint8
    )  # Create a mask for the reference image
    canvas = cv2.add(canvas, ref_warped, mask=mask_ref.max(axis=2))

    # Step 4: Warp and blend all other images into the canvas
    for node in image_nodes:
        if node.homography is not None:
            # Combine the translation and the homography for the current image
            combined_homography = translation_matrix @ node.homography

            # Warp the image into the canvas
            warped_img = cv2.warpPerspective(
                node.image, combined_homography, canvas_size
            )

            # Create a mask for the current image
            mask_img = (warped_img > 0).astype(np.uint8)

            # Add the warped image to the canvas

            # Normalize the images for blending
            canvas_float = canvas.astype(np.float32) / 255.0
            warped_float = warped_img.astype(np.float32) / 255.0

            # Blend the images together using maximum intensity projection or simple overlay
            for c in range(3):  # Loop over each channel
                canvas_channel = canvas_float[:, :, c]
                warped_channel = warped_float[:, :, c]

                # Blend the two images using the max of both
                canvas_channel = np.maximum(canvas_channel, warped_channel)
                canvas[:, :, c] = (canvas_channel * 255).astype(np.uint8)

    return canvas


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


def show_inlier_matches(
    node1, node2, max_desc_dist=None, ransac_threshold=RANSAC_THRESHOLD, max_points=None
):
    """
    Finds matches between two image nodes, computes a homography, and
    visualizes only the inlier matches.

    Args:
        node1 (ImageNode): The first image node (has .image, .keypoints, .descriptors).
        node2 (ImageNode): The second image node.
        max_desc_dist (float, optional): Maximum descriptor distance to keep a match.
        ransac_threshold (float): Threshold for RANSAC homography.
        max_points (int or None): If int, draw up to this many inlier points.
                                  If None, draw all inliers.
    """
    # 1) Find initial descriptor matches + their points
    src_pts, dst_pts = find_matches_numpy(node1, node2, max_desc_dist)

    # 2) Compute homography and get RANSAC's inlier mask
    H, mask = compute_homography_ransac(src_pts, dst_pts, ransac_threshold)
    if mask is None:
        print("No homography could be computed.")
        return

    # 3) Filter only inlier points
    inlier_indices = np.where(mask.flatten() == 1)[0]
    inlier_src_pts = src_pts[inlier_indices]
    inlier_dst_pts = dst_pts[inlier_indices]

    # 4) Draw the inlier matches in a side-by-side image
    out_image = draw_matches_points(
        node1.image, node2.image, inlier_src_pts, inlier_dst_pts, max_points=max_points
    )

    # 5) Show the result
    plt.figure(figsize=(12, 6))
    plt.imshow(out_image)
    plt.title(f"Inlier Matches between Node {node1.idx} and Node {node2.idx}")
    plt.axis("off")
    plt.show()
