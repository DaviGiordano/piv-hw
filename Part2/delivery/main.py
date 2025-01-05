from ImageNode import ImageNode
from ImageNodeMatch import ImageNodesMatch
from PoseGraph import PoseGraph
from pathlib import Path
import scipy.io as spio
import open3d as o3d
import os
import numpy as np


def parse_cams_info(cams_info_raw):
    cams_info = []
    for datapoint in cams_info_raw:
        img = datapoint[0][0][0][0]
        depth_map = datapoint[0][0][0][1]
        conf_map = datapoint[0][0][0][2]
        focal_length = datapoint[0][0][0][3][0][0]
        cams_info.append(
            {
                "focal_length": focal_length,
                "rgb": img,
                "depth": depth_map,
                "conf": conf_map,
            }
        )
    return cams_info


def parse_sift(sift_raw):

    keys = list(set(sift_raw.keys()) - {"__header__", "__version__", "__globals__"})
    keys = sorted(keys, key=lambda x: int(x.split("_")[1][3:]))
    sift = []
    for key in keys:
        sift.append({"kp": sift_raw[key][0][0][0], "desc": sift_raw[key][0][0][1]})

    return sift


def create_image_nodes(cams_info, sift):

    image_nodes: ImageNode = []

    for i in range(len(cams_info)):
        image_nodes.append(
            ImageNode(
                idx=i,
                rgb=cams_info[i]["rgb"],
                keypoints=sift[i]["kp"],
                descriptors=sift[i]["desc"],
                depth_map=cams_info[i]["depth"],
                conf_map=cams_info[i]["conf"],
                focal_length=cams_info[i]["focal_length"],
            )
        )
    return image_nodes


def get_reference_node(num_nodes):
    while True:
        try:
            ref_idx = int(
                input(f"Enter the reference image index (0 to {num_nodes - 1}): ")
            )
            if 0 <= ref_idx < num_nodes:
                return ref_idx
            else:
                print(f"Error: Please enter a number between 0 and {num_nodes - 1}.")
        except ValueError:
            print("Error: Invalid input. Please enter an integer.")


def get_included_images(num_nodes):
    while True:
        try:
            included_images = input(
                f"Enter the image indexes to include in the reconstruction, separated by commas (0 to {num_nodes - 1}): "
            )
            indexes = [int(idx.strip()) for idx in included_images.split(",")]
            if all(0 <= idx < num_nodes for idx in indexes):
                return indexes
            else:
                print(f"Error: All indexes must be between 0 and {num_nodes - 1}.")
        except ValueError:
            print("Error: Invalid input. Please enter integers separated by commas.")


def filter_included_nodes(image_nodes, included_images):
    return [node for node in image_nodes if node.idx in included_images]


def get_image_nodes_dict(image_nodes):
    image_nodes_dict = {}
    for node in image_nodes:
        image_nodes_dict.update({node.idx: node})
    return image_nodes_dict


def apply_transform(points, transform):
    """
    Applies a 4x4 transform to (N,3) points. Returns (N,3).
    """
    N = points.shape[0]
    hom_pts = np.hstack([points, np.ones((N, 1))])  # (N,4)
    transformed = (transform @ hom_pts.T).T  # (N,4)
    return transformed[:, :3]


# Point cloud functions
def validate_and_stack_points(all_points, all_colors, pcloud_transf, cloud_colors, idx):
    if len(pcloud_transf) != len(cloud_colors):
        raise ValueError(
            f"Mismatch between point cloud and color array sizes for node {idx}: "
            f"{len(pcloud_transf)} points, {len(cloud_colors)} colors."
        )
    all_points.append(pcloud_transf)
    all_colors.append(cloud_colors)


def save_combined_point_cloud(all_points, all_colors, output_path):
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    pcd = register_point_cloud(all_points, all_colors)
    combined_output_fpath = output_path / "combined_point_cloud.ply"
    write_point_cloud(pcd, combined_output_fpath)


def register_point_cloud(points, cloud_colors) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(cloud_colors)
    return pcd


def write_point_cloud(pcd, output_fpath):
    o3d.io.write_point_cloud(str(output_fpath), pcd, write_ascii=True)
    print(f"Point cloud saved to {output_fpath}")


def main():
    ## Configure dirs
    script_dir = Path(__file__).parent
    data_path = script_dir
    output_path = script_dir

    cams_info_fpath = (
        data_path / "cams_info.mat"
    )  # ! Originally named "cams_info_no_extr.mat" in the office dataset
    sift_path = data_path / "kp.mat"

    ## Load data
    cams_info_raw = spio.loadmat(cams_info_fpath)["cams_info"]
    cams_info = parse_cams_info(cams_info_raw)

    sift_raw = spio.loadmat(sift_path)
    sift = parse_sift(sift_raw)

    image_nodes = create_image_nodes(cams_info, sift)

    print(f">> {len(image_nodes)} images have been loaded.")

    ## Input user
    reference_node = get_reference_node(len(image_nodes))
    included_images = get_included_images(len(image_nodes))

    # Filter nodes
    included_images = list(set(included_images + [reference_node]))
    image_nodes = filter_included_nodes(image_nodes, included_images)
    image_nodes_dict = get_image_nodes_dict(image_nodes)

    # Match nodes
    nodes_match = {}
    for node1 in image_nodes:
        for node2 in image_nodes:
            matching_nodes = ImageNodesMatch(node1, node2)
            nodes_match.update({matching_nodes.pair_idx: matching_nodes})

    # Register point cloud
    optimizer = PoseGraph(image_nodes, nodes_match)
    optimizer.incremental_registration(base_node_idx=reference_node)

    all_points, all_colors = [], []
    for idx, transform in optimizer.global_poses.items():
        pcloud_transf = apply_transform(image_nodes_dict[idx].point_cloud, transform)
        validate_and_stack_points(
            all_points,
            all_colors,
            pcloud_transf,
            image_nodes_dict[idx].cloud_colors,
            idx,
        )

    save_combined_point_cloud(all_points, all_colors, output_path)


if __name__ == "__main__":
    main()
