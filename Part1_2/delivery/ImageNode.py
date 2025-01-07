import numpy as np


class ImageNode:

    def __init__(
        self,
        image,
        keypoints,
        descriptors,
        idx,
        input_fpath,
        output_path,
        yolo_detections,
    ):
        self.image = image
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.idx = idx

        self.footprint = [idx]
        self.input_fpath = input_fpath
        self.output_path = output_path
        self.yolo_detections = yolo_detections
        self.image_num = None
        self.homography = None
        self.yolo_transformed = None

    def __str__(self):
        return f"ImageNode(idx={self.idx}, keypoints_shape={self.keypoints.shape}, descriptors_shape={self.descriptors.shape}, footprint={self.footprint})"

    def transform_yolo_detections(self):
        if self.homography is None:
            print(f"Node {self.idx} has no homography. Yolo was not transformed")
            return
        if self.yolo_detections is None:
            print(f"Node {self.idx} has no yolo detections. Yolo was not computed")
            return

        transformed = []
        for x1, y1, x2, y2 in self.yolo_detections["xyxy"]:
            c1 = self.homography @ np.array([x1, y1, 1.0])
            c1 /= c1[2]
            c2 = self.homography @ np.array([x2, y2, 1.0])
            c2 /= c2[2]
            transformed.append([c1[0], c1[1], c2[0], c2[1]])

        self.yolo_transformed = self.yolo_detections.copy()
        self.yolo_transformed["xyxy"] = np.array(transformed)
