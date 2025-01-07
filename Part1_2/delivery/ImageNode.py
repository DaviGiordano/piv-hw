import cv2
import matplotlib.pyplot as plt
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
        return f"ImageNode(idx={self.id}, keypoints_shape={self.keypoints.shape}, descriptors_shape={self.descriptors.shape}, footprint={self.footprint})"

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

    def show_original_image_with_yolo(self):
        """
        Displays the original image with YOLO bounding boxes (before transformation).
        """
        if self.yolo_detections is None:
            print(f"Node {self.idx} has no YOLO detections.")
            return

        # Copy the original image to avoid modifying it
        image_with_boxes = self.image.copy()

        # Draw YOLO bounding boxes
        for x1, y1, x2, y2 in self.yolo_detections["xyxy"]:
            cv2.rectangle(
                image_with_boxes,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),  # Green color for boxes
                2,  # Thickness
            )

        # Display the image with bounding boxes
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
        plt.title(f"Original Image - Node {self.idx}")
        plt.axis("off")
        plt.show()


def get_warped_image_with_yolo(self):
    """
    Returns a copy of this node's image warped to the reference frame (via 'self.homography'),
    with transformed YOLO bounding boxes drawn on it.
    """
    if self.homography is None:
        return None

    # Warp the image
    h, w = self.image.shape[:2]
    warped = cv2.warpPerspective(self.image, self.homography, (w, h))

    # Draw YOLO boxes if they exist
    if self.yolo_transformed is not None:
        for x1, y1, x2, y2 in self.yolo_transformed["xyxy"]:
            cv2.rectangle(
                warped, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
            )

    return warped
