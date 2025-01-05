import numpy as np

MIN_CONF = 0.5


class ImageNode:
    def __init__(
        self, idx, rgb, keypoints, descriptors, depth_map, conf_map, focal_length
    ):
        self.rgb = rgb
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.idx = idx
        self.depth_map = depth_map
        self.conf_map = conf_map
        self.focal_length = focal_length

        self.camera_intrinsic = self.get_camera_intrinsic()
        self.point_cloud = self.get_point_cloud()
        self.cloud_colors = self.get_point_cloud_colors()

        # Filters keypoints with confidence maps
        self.keypoints, self.descriptors = self.filter_2d_keypoints_and_descriptors()
        self.keypoints_3d = self.get_3d_keypoints()

    def __str__(self):
        return (
            f"ImageNode(idx={self.idx}, "
            f"focal_length={self.focal_length}, "
            f"rgb_shape={self.rgb.shape}, "
            # f"keypoints_shape={self.keypoints.shape}, "
            # f"descriptors_shape={self.descriptors.shape}, "
            f"depth_map_shape={self.depth_map.shape}, "
            f"conf_map_shape={self.conf_map.shape})"
        )

    def get_camera_intrinsic(self):
        h, w = self.rgb.shape[:2]
        cx, cy = w / 2, h / 2
        intrinsic_matrix = np.array(
            [[self.focal_length, 0, cx], [0, self.focal_length, cy], [0, 0, 1]]
        )
        return intrinsic_matrix

    def get_point_cloud(self, min_conf=MIN_CONF):
        """
        Converts depth map to 3D point cloud, discarding points
        with confidence < min_conf.
        """
        h, w = self.rgb.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Flatten
        pixel_coords_flat = np.stack(
            (x.ravel(), y.ravel(), np.ones_like(x.ravel())), axis=0
        )
        depth_flat = self.depth_map.ravel()
        conf_flat = self.conf_map.ravel()

        # Discard low-confidence points
        valid_mask = conf_flat >= min_conf
        pixel_coords_flat_valid = pixel_coords_flat[:, valid_mask]
        depth_flat_valid = depth_flat[valid_mask]

        # Scale pixel coordinates by depth
        pixels_scaled = pixel_coords_flat_valid * depth_flat_valid

        # Transform to 3D using the inverse of the camera intrinsic
        pcloud_flat = np.linalg.inv(self.get_camera_intrinsic()) @ pixels_scaled

        return pcloud_flat.T  # (N, 3)

    def filter_2d_keypoints_and_descriptors(self, min_conf=MIN_CONF):
        """
        Filter out keypoints & descriptors based on the confidence map.
        Returns filtered keypoints and descriptors.
        """
        h, w = self.depth_map.shape[:2]
        filtered_kpts = []
        filtered_desc = []

        for kp, desc in zip(self.keypoints, self.descriptors):
            x2d, y2d = map(int, kp)

            # Check image bounds
            if not (0 <= x2d < w and 0 <= y2d < h):
                raise ValueError(f"Keypoint ({x2d, y2d}) out of bounds.")

            # Check confidence threshold
            if self.conf_map[y2d, x2d] >= min_conf:
                filtered_kpts.append(kp)
                filtered_desc.append(desc)

        # Convert filtered_desc back to a NumPy array if it was originally one
        if len(filtered_desc) > 0:
            filtered_desc = np.vstack(filtered_desc)
        else:
            raise ValueError("No keypoints found with above confidence threshold.")

        return filtered_kpts, filtered_desc

    def get_high_conf_img(self, min_conf=MIN_CONF):
        # Ensure shapes match
        assert (
            self.rgb.shape[:2] == self.conf_map.shape
        ), "self.rgb and self.conf_map must have the same height and width"

        # Filter by min_conf
        filtered_image = self.rgb.copy()
        high_conf_mask = self.conf_map >= min_conf
        filtered_image[~high_conf_mask] = [0, 0, 0]

        return filtered_image

    def get_3d_keypoints(self):
        """
        Back-projects 2D keypoints to 3D using the depth map.
        Returns an (N, 3) array, where N = len(keypoints).
        """
        K_inv = np.linalg.inv(self.get_camera_intrinsic())
        h, w = self.depth_map.shape[:2]

        keypoints_3d = []
        for kp in self.keypoints:
            x2d, y2d = map(int, kp)
            if not (0 <= x2d < w and 0 <= y2d < h):
                raise ValueError(f"Keypoint ({x2d, y2d}) out of bounds.")

            depth_val = self.depth_map[y2d, x2d]
            pixel_hom = np.array([x2d * depth_val, y2d * depth_val, depth_val])
            xyz = K_inv @ pixel_hom
            keypoints_3d.append(xyz)

        return np.array(keypoints_3d, dtype=np.float32)

    def get_point_cloud_colors(self, min_conf=MIN_CONF):
        """
        Unravels the RGB image to an (N, 3) matrix where N is the number of valid 3D points,
        filtered using the confidence map.
        """
        h, w = self.rgb.shape[:2]
        conf_flat = self.conf_map.ravel()
        rgb_flat = self.rgb.reshape((h * w, 3))

        # Filter colors by confidence
        valid_mask = conf_flat >= min_conf
        colors = rgb_flat[valid_mask]

        return colors
