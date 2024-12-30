import cv2

class ImageNode:
    def __init__(self, image, keypoints, descriptors, idx):
        self.image = image
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.idx = idx
        self.homography = None
        self.footprint = [idx]

    def __str__(self):
        return f"ImageNode(idx={self.idx}, keypoints_shape={self.keypoints.shape}, descriptors_shape={self.descriptors.shape}, footprint={self.footprint})"

def resize_keypoint_and_image(keypoint, image, scale_factor):

    resized_keypoint = keypoint * scale_factor
    resized_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    
    return resized_keypoint, resized_image

def print_footprints(image_nodes):
    for node in image_nodes:
        print(node.footprint)