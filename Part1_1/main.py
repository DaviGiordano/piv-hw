from utils import warp_image, load_img, transform_points, compute_homography
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
import scipy as sp
import logging
import json
import os

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_output_directory(output_path):
    output_path.mkdir(parents=True, exist_ok=True)

def load_data_paths(data_path):
    yolo_fpaths = sorted(data_path.glob('yolo*.mat'), key=lambda x: int(x.stem.split('_')[1]))
    img_fpaths = sorted(data_path.glob('img*.jpg'), key=lambda x: int(x.stem.split('_')[1]))
    assert len(img_fpaths) == len(yolo_fpaths),\
        "Number of images and YOLO files do not match."
    for i in range(len(img_fpaths)):
        assert img_fpaths[i].stem.split('_')[1] == yolo_fpaths[i].stem.split('_')[1],\
        f"Image and YOLO file order mismatch at index {i}."
    return img_fpaths, yolo_fpaths

def load_keypoints(data_path):
    kp_gmaps_fpath = data_path / 'kp_gmaps.mat'
    matches = sp.io.loadmat(kp_gmaps_fpath)['kp_gmaps']
    kp_video = matches[:, :2]
    kp_map = matches[:, 2:]
    return kp_video, kp_map

def save_homography(H, output_path):
    sp.io.savemat(output_path / 'homography.mat', {'H': H})
    logging.info(f'Homography saved to {output_path}')

def transform_yolo_coordinates(yolo_data, H):
    bounding_boxes = yolo_data['xyxy']
    transformed_bounding_boxes = []
    for bbox in bounding_boxes:
        x_min, y_min, x_max, y_max = bbox
        points = np.array([[x_min, y_min], [x_max, y_max]])
        transformed_points = transform_points(points, H)
        x_min_transformed, y_min_transformed = transformed_points[0]
        x_max_transformed, y_max_transformed = transformed_points[1]
        transformed_bounding_boxes.append([x_min_transformed, y_min_transformed, x_max_transformed, y_max_transformed])
    return np.array(transformed_bounding_boxes)

def plot_transformed_yolo_data(img_warped, transformed_bounding_boxes, output_img_fpath):
    fig, ax = plt.subplots(1)
    ax.imshow(img_warped)
    for bbox in transformed_bounding_boxes:
        x_min, y_min, x_max, y_max = bbox
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.savefig(output_img_fpath)
    plt.close(fig)

def process_images_and_yolo(img_fpaths, yolo_fpaths, H, output_path, shape):
    for img_fpath, yolo_fpath in zip(img_fpaths, yolo_fpaths):
        logging.info(f"Processing {img_fpath.name} and {yolo_fpath.name}..")
        yolo_data = sp.io.loadmat(yolo_fpath)
        transformed_bounding_boxes = transform_yolo_coordinates(yolo_data, H)
        yolo_data['xyxy'] = transformed_bounding_boxes
        output_yolo_fpath = output_path / yolo_fpath.name
        sp.io.savemat(output_yolo_fpath, yolo_data)
        img = load_img(img_fpath)
        img_warped = warp_image(img, H, shape)
        plot_transformed_yolo_data(img_warped, transformed_bounding_boxes, output_path / img_fpath.name)
        logging.info(f'Output saved to {output_path}')

def main():
    setup_logging()
    data_path = Path(os.getcwd())
    output_path = Path(os.getcwd()) / 'output'
    create_output_directory(output_path)
    img_fpaths, yolo_fpaths = load_data_paths(data_path)
    kp_video, kp_map = load_keypoints(data_path)
    H = compute_homography(kp_video, kp_map)
    save_homography(H, output_path)
    shape = (1980, 1590)
    process_images_and_yolo(img_fpaths, yolo_fpaths, H, output_path, shape)

if __name__ == "__main__":
    main()
