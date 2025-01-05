import numpy as np


class PoseGraph:
    def __init__(self, image_nodes, nodes_match):
        """
        Args:
            image_nodes (list[ImageNode]): All ImageNodes.
            nodes_match (dict): Dictionary of ((idx1, idx2) -> ImageNodesMatch).
        """
        self.image_nodes = image_nodes
        self.nodes_match = nodes_match

        # Step 1: Build a graph of edges (src_idx, dst_idx, transform, error)
        self.edges = self._build_pose_graph()

        # Initialize each node's global transform as identity
        self.global_poses = {n.idx: np.eye(4) for n in self.image_nodes}

    def _build_pose_graph(self):
        """
        Build a list of edges of the form:
            (nodeA_idx, nodeB_idx, transform, error),
        using the ICP transform and its error.
        """
        edges = []
        for (i, j), match in self.nodes_match.items():
            if i != j:
                transform_icp = match.transform_icp  # 4x4 transform
                error_icp = match.get_error(transform_type="icp")
                edges.append((i, j, transform_icp, error_icp))
        return edges

    def incremental_registration(self, base_node_idx=0):
        """
        Step 2:
          - Sort edges by registration error.
          - Start with 'base_node_idx' as the root (pose = identity).
          - Incrementally add nodes by picking edges with the lowest error
            that connect a new node to the current set.
        """
        # Sort edges by ascending error
        self.edges.sort(key=lambda e: e[3])

        visited = set([base_node_idx])
        remaining = set([n.idx for n in self.image_nodes]) - visited

        # Build a spanning tree incrementally
        while remaining:
            print(f"Node(s) {remaining} remaining")
            for node1_idx, node2_idx, transform, err in self.edges:
                # If this edge connects a visited node to an unvisited node

                if node2_idx in visited and node1_idx in remaining:
                    # Compose transformation
                    pose = self.global_poses[node2_idx]
                    self.global_poses[node1_idx] = transform @ pose
                    visited.add(node1_idx)
                    remaining.remove(node1_idx)
                    print(f"Transformed {node1_idx} to {node2_idx}")
                    break

                elif node1_idx in visited and node2_idx in remaining:
                    pose = self.global_poses[node1_idx]
                    self.global_poses[node2_idx] = np.linalg.inv(transform) @ pose
                    visited.add(node2_idx)
                    remaining.remove(node2_idx)
                    print(f"Transformed {node2_idx} to {node1_idx}")
                    break
