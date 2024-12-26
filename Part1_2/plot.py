from matplotlib import pyplot as plt
import numpy as np
import cv2

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
        corners_ref = np.array([
            [0, 0],
            [w_ref, 0],
            [w_ref, h_ref],
            [0, h_ref]
        ], dtype=np.float32).reshape(-1, 1, 2)

        corners_warp = np.array([
            [0, 0],
            [w_warp, 0],
            [w_warp, h_warp],
            [0, h_warp]
        ], dtype=np.float32).reshape(-1, 1, 2)

        # Warp the corners of the current image to the reference frame
        warped_corners = cv2.perspectiveTransform(corners_warp, H)

        # Combine corners to find the overall bounds
        all_corners = np.concatenate((corners_ref, warped_corners), axis=0)

        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        # Compute the translation needed to shift the images
        translation = [-xmin, -ymin]

        # Define the translation matrix with float32 type
        translation_matrix = np.array([
            [1, 0, translation[0]],
            [0, 1, translation[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Warp the reference image using the translation matrix
        ref_warped = cv2.warpPerspective(ref_img, translation_matrix, (xmax - xmin, ymax - ymin))

        # Warp the current image using the combined translation and homography matrices
        combined_homography = translation_matrix @ H  # Matrix multiplication
        warped_img = cv2.warpPerspective(node.image, combined_homography, (xmax - xmin, ymax - ymin))

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
        axes[0].imshow(ref_warped, cmap='gray' if ref_warped.ndim == 2 else None)
        axes[0].set_title("Reference Image (Warped to Canvas)")
        axes[0].axis('off')

        # Show the warped current image
        axes[1].imshow(warped_img, cmap='gray' if warped_img.ndim == 2 else None)
        axes[1].set_title(f"Warped Image (idx={node.idx})")
        axes[1].axis('off')

        # Show the overlay
        axes[2].imshow(overlay, cmap='gray' if overlay.ndim == 2 else None)
        axes[2].set_title("Overlay (Ref & Warped)")
        axes[2].axis('off')

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
    corners_ref = np.array([
        [0, 0],
        [w_ref, 0],
        [w_ref, h_ref],
        [0, h_ref]
    ], dtype=np.float32).reshape(-1, 1, 2)
    
    all_corners = [corners_ref]
    
    for node in image_nodes:
        if node.homography is not None:
            h_img, w_img = node.image.shape[:2]
            corners_img = np.array([
                [0, 0],
                [w_img, 0],
                [w_img, h_img],
                [0, h_img]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
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
    translation_matrix = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Step 3: Warp the reference image into the new canvas
    canvas_size = (xmax - xmin, ymax - ymin)
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)  # Create an empty canvas
    
    # Warp the reference image
    ref_warped = cv2.warpPerspective(ref_img, translation_matrix, canvas_size)
    
    # Add the reference image to the canvas
    mask_ref = (ref_warped > 0).astype(np.uint8)  # Create a mask for the reference image
    canvas = cv2.add(canvas, ref_warped, mask=mask_ref.max(axis=2))
    
    # Step 4: Warp and blend all other images into the canvas
    for node in image_nodes:
        if node.homography is not None:
            # Combine the translation and the homography for the current image
            combined_homography = translation_matrix @ node.homography
            
            # Warp the image into the canvas
            warped_img = cv2.warpPerspective(node.image, combined_homography, canvas_size)
            
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
