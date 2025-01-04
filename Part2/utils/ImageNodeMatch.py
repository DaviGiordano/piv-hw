from scipy.spatial import cKDTree
import numpy as np
import cv2

RANSAC_THRESHOLD = 5


class ImageNodesMatch:
    def __init__(self, src_node, dst_node):
        self.src_node = src_node
        self.dst_node = dst_node
        self.pair_idx = (src_node.idx, dst_node.idx)

        self.matches = self.get_keypoint_matches()

        self.src_2d, self.dst_2d = self.get_2d_match_pts()
        self.src_3d, self.dst_3d = self.get_3d_match_pts()

        # Transformation matrices
        self.transform_procrustes = self.get_rigid_transform_procrustes()
        self.transform_procrustes_ransac = self.get_rigid_transform_procrustes_ransac()
        self.transform_icp = self.refine_transform_icp()

    def get_keypoint_matches(self):
        """
        Match keypoints & descriptors between src_node and dst_node.
        """
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(self.src_node.descriptors, self.dst_node.descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def get_2d_match_pts(self):
        """
        Retrieve matched 2D coordinates.
        """
        if self.matches is None:
            raise ValueError(
                "Matches not computed yet. Call get_keypoint_matches first."
            )

        src_pts = []
        dst_pts = []
        for m in self.matches:
            kp1 = self.src_node.keypoints[m.queryIdx]
            kp2 = self.dst_node.keypoints[m.trainIdx]
            src_pts.append(kp1)
            dst_pts.append(kp2)

        return np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32)

    def get_3d_match_pts(self):
        """
        Given two ImageNode objects (node1, node2) and a list of OpenCV-style matches,
        returns two arrays of corresponding 3D points.

        Assumes the node1/node2.keypoints & descriptors are already filtered
        if needed (so 'matches' aligns with these filtered sets).
        """
        # 1) Convert all node1's and node2's keypoints to 3D

        # 2) Collect matching 3D points
        src_3d = []
        dst_3d = []
        for m in self.matches:
            p1 = self.src_node.keypoints_3d[m.queryIdx]
            p2 = self.dst_node.keypoints_3d[m.trainIdx]

            src_3d.append(p1)
            dst_3d.append(p2)

        return np.array(src_3d, dtype=np.float32), np.array(dst_3d, dtype=np.float32)

    def get_rigid_transform_procrustes(self):
        """
        Compute the Procrustes-based rigid transform (R, t) from src_3d to dst_3d,
        without any outlier rejection.
        """
        if self.src_3d is None or self.dst_3d is None:
            raise ValueError("3D points not ready.")

        return self._estimate_transform_procrustes(self.src_3d, self.dst_3d)

    def get_rigid_transform_procrustes_ransac(
        self, max_iterations=1000, sample_size=3, inlier_threshold=RANSAC_THRESHOLD
    ):
        """
        Compute a robust Procrustes transform using RANSAC on 3D-3D correspondences.
        """
        if self.src_3d is None or self.dst_3d is None:
            raise ValueError("3D points not ready.")

        if self.src_3d.shape[0] != self.dst_3d.shape[0]:
            raise ValueError(
                "Source and destination must have the same number of points."
            )

        src_3d = self.src_3d
        dst_3d = self.dst_3d
        n_points = src_3d.shape[0]

        best_inlier_count = -1
        best_transform = np.eye(4)
        best_inliers = np.zeros(n_points, dtype=bool)

        for _ in range(max_iterations):
            # 1) Randomly sample 'sample_size' correspondences
            indices = np.random.choice(n_points, sample_size, replace=False)
            src_sample = src_3d[indices]
            dst_sample = dst_3d[indices]

            # 2) Estimate transform from the sample
            transform_candidate = self._estimate_transform_procrustes(
                src_sample, dst_sample
            )

            # 3) Count inliers
            inliers_mask = self._compute_inliers(
                src_3d, dst_3d, transform_candidate, inlier_threshold
            )
            inlier_count = np.sum(inliers_mask)

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_transform = transform_candidate
                best_inliers = inliers_mask

        # (Optional) Refine using all inliers from the best model
        if best_inlier_count > 0:
            src_inliers = src_3d[best_inliers]
            dst_inliers = dst_3d[best_inliers]
            best_transform = self._estimate_transform_procrustes(
                src_inliers, dst_inliers
            )
        self.src_3d = self.src_3d[best_inliers]
        self.dst_3d = self.dst_3d[best_inliers]
        self.src_2d = self.src_2d[best_inliers]
        self.dst_2d = self.dst_2d[best_inliers]

        return best_transform

    def refine_transform_icp(self, max_iterations=20, tolerance=1e-5):
        """
        Refine the transform with a simple ICP approach:

        1) Start with the Procrustes transform as an initial guess.
        2) For each iteration:
           a) Transform src_3d with the current guess.
           b) Find nearest neighbors in dst_3d.
           c) Use Procrustes on the matched pairs to get a new transform.
           d) Compose the new transform with the old one.
           e) Check if the update is small enough to stop.
        """
        if self.src_3d is None or self.dst_3d is None:
            raise ValueError("3D points not ready.")

        # Initial guess from Procrustes (or RANSAC)
        transform = self.transform_procrustes_ransac.copy()

        src_pts = self.src_3d
        dst_pts = self.dst_3d

        # Build KD-tree for nearest neighbor queries on the destination points
        dst_kdtree = cKDTree(dst_pts)

        for _ in range(max_iterations):
            # 1) Transform src_3d with the current guess
            src_aligned = self.apply_transform(src_pts, transform)

            # 2) Find nearest neighbors in dst_3d
            #    distances, idx[i] -> the index of the nearest neighbor for src_aligned[i]
            distances, nn_indices = dst_kdtree.query(src_aligned)

            # 3) Build matched pairs: (src_aligned[i], dst_pts[idx[i]])
            matched_src = src_aligned
            matched_dst = dst_pts[nn_indices]

            # 4) Compute new transform from these matched pairs
            new_transform = self._estimate_transform_procrustes(
                matched_src, matched_dst
            )

            # 5) Compose new_transform with the current transform
            old_transform = transform.copy()
            transform = new_transform @ transform  # combine transforms

            # 6) Check if change is small enough to converge
            diff = np.linalg.norm(transform - old_transform)
            if diff < tolerance:
                break

        self.transform_icp = transform
        return transform

    def get_error(self, transform_type="procrustes"):
        """
        Compute a simple alignment error (e.g., mean Euclidean distance)
        based on the chosen transform.
        """
        if transform_type == "procrustes":
            transform = self.transform_procrustes
        elif transform_type == "procrustes_ransac":
            transform = self.transform_procrustes_ransac
        elif transform_type == "icp":
            transform = self.transform_icp
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

        if transform is None:
            raise ValueError(f"{transform_type} transform not computed yet.")

        # Transform src_3d
        src_h = np.hstack([self.src_3d, np.ones((self.src_3d.shape[0], 1))])
        src_transformed = (transform @ src_h.T).T[:, :3]

        # Compute mean error
        errors = np.linalg.norm(src_transformed - self.dst_3d, axis=1)
        return np.mean(errors)

    def apply_transform(self, points, transform):
        """
        Applies a 4x4 transform to (N,3) points. Returns (N,3).
        """
        N = points.shape[0]
        hom_pts = np.hstack([points, np.ones((N, 1))])  # (N,4)
        transformed = (transform @ hom_pts.T).T  # (N,4)
        return transformed[:, :3]

    @staticmethod
    def _estimate_transform_procrustes(src_pts, dst_pts):
        """
        Given two sets of 3D correspondences (src_pts, dst_pts),
        compute the Procrustes transform (4x4) that aligns src_pts to dst_pts.
        """
        src_mean = np.mean(src_pts, axis=0)
        dst_mean = np.mean(dst_pts, axis=0)

        src_centered = src_pts - src_mean
        dst_centered = dst_pts - dst_mean

        H = src_centered.T @ dst_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = dst_mean - R @ src_mean

        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        return transform

    def _compute_inliers(self, src_pts, dst_pts, transform, threshold=RANSAC_THRESHOLD):
        """
        Transforms 'src_pts' by 'transform', then computes distances to 'dst_pts'.
        Returns a boolean mask of inliers (distance < threshold).
        """
        src_aligned = self.apply_transform(src_pts, transform)
        distances = np.linalg.norm(src_aligned - dst_pts, axis=1)
        return distances < threshold
