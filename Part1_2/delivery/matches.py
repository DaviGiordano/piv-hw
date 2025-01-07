import numpy as np


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
            np.sum(src_descriptors[:, None, :] ** 2, axis=2)
            + np.sum(dst_descriptors[None, :, :] ** 2, axis=2)
            - 2 * np.dot(src_descriptors, dst_descriptors.T),
        )
    )

    # Find the nearest neighbor in both directions
    src_to_dst = np.argmin(distances, axis=1)
    dst_to_src = np.argmin(distances, axis=0)

    # Reciprocal matching (cross-check logic)
    reciprocal_matches = [
        (src_idx, dst_idx)
        for src_idx, dst_idx in enumerate(src_to_dst)
        if dst_to_src[dst_idx] == src_idx
    ]

    # Apply distance threshold if provided
    if max_desc_dist is not None:
        reciprocal_matches = [
            (src_idx, dst_idx)
            for src_idx, dst_idx in reciprocal_matches
            if distances[src_idx, dst_idx] <= max_desc_dist
        ]

    # Extract matched keypoints
    src_pts = np.array(
        [src_node.keypoints[src_idx] for src_idx, _ in reciprocal_matches],
        dtype=np.float32,
    )
    dst_pts = np.array(
        [dst_node.keypoints[dst_idx] for _, dst_idx in reciprocal_matches],
        dtype=np.float32,
    )

    return src_pts, dst_pts
