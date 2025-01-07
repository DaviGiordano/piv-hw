import os
import numpy as np
import scipy.io as sio
from collections import defaultdict


def save_homographies_and_yolo(image_nodes):
    # Group nodes by their output folder
    groups = defaultdict(list)
    for node in image_nodes:
        groups[node.output_path].append(node)

    # For each output folder, save homographies and YOLO detections
    for out_dir, nodes in groups.items():
        os.makedirs(out_dir, exist_ok=True)

        # 1) Save homographies in a (3,3,Nv) array
        homographies = []
        for node in nodes:
            if node.homography is None:
                raise ValueError(f"Homography for node {node} is None")
            homographies.append(node.homography)
        H_array = np.stack(homographies, axis=-1)  # shape (3, 3, Nv)

        homography_path = os.path.join(out_dir, "homographies.mat")
        sio.savemat(homography_path, {"H": H_array})

        # 2) Save transformed YOLO detections in individual files
        for i, node in enumerate(nodes):
            fname = f"yolooutput_{int(node.idx.split('_')[-1]):04d}.mat"
            out_path = os.path.join(out_dir, fname)
            if node.yolo_transformed:
                sio.savemat(out_path, node.yolo_transformed)
            else:
                print(f"Node {node.idx} has no yolo")
