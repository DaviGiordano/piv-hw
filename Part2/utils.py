import numpy as np

class ImageNode:
    def __init__(self, idx, rgb, keypoints, descriptors, depth_map, conf_map, focal_length):
        self.rgb = rgb
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.idx = idx
        self.depth_map = depth_map
        self.conf_map = conf_map
        self.focal_length = focal_length

        self.camera_intrinsic = None
        self.point_cloud = None
        
    def __str__(self):
        return (f"ImageNode(idx={self.idx}, "
                f"focal_length={self.focal_length}, "
                f"rgb_shape={self.rgb.shape}, "
                f"keypoints_shape={self.keypoints.shape}, "
                f"descriptors_shape={self.descriptors.shape}, "
                f"depth_map_shape={self.depth_map.shape}, "
                f"conf_map_shape={self.conf_map.shape})")
    

    def get_camera_intrinsic(self):
        h, w = self.rgb.shape[:2]
        cx, cy = w / 2, h / 2
        intrinsic_matrix = np.array([
            [self.focal_length, 0, cx],
            [0, self.focal_length, cy],
            [0, 0, 1]
        ])
        return intrinsic_matrix
    
    def get_point_cloud(self):
        h, w = self.rgb.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        pixel_coords_flat = np.stack((x.ravel(), y.ravel(), np.ones_like(x.ravel())), axis=0)  # Shape: (3, N)
        depth_flat = self.depth_map.ravel()  # Shape: (N,)

        pixels_scaled = pixel_coords_flat * depth_flat  # (3, N) * (N,) -> (3, N)
        pcloud_flat = np.linalg.inv(self.camera_intrinsic) @ pixels_scaled  # (3, 3) @ (3, N) -> (3, N)
        return pcloud_flat.T # (N, 3)
