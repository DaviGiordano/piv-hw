# Part 2: Point Cloud Registration and Reconstruction

This repository provides a pipeline for reconstructing and registering 3D point clouds from a set of images. Adjust `MIN_CONF` in `ImageNode`  or `RANSAC_THRESHOLD` in `ImageNodeMatch` as desired (Default: `MIN_CONF`=$0.7$;  `RANSAC_THRESHOLD`=$5$).

## Dependencies
- Python 3.7 or later
- Required libraries:
  - `numpy`
  - `scipy`
  - `open3d`
  - `opencv`

## Usage

### Input Files
- `cams_info.mat`: MATLAB struct file containing camera parameters (`rgb`, `depth_map`, `conf_map`, `focal_length`).
- `kp.mat`: MATLAB file containing SIFT keypoints and descriptors.

The parsing of these files was based on the file `cams_info_no_extr.mat` made available in the PIV Shared Drive. See functions `parse_cams_info` and `parse_sift` in `main.py`

### Running the Script
To run the reconstruction pipeline, execute the `main.py` script.

The script will prompt you for:
1. Reference Image Index: Choose an image to use as the reference for the incremental registration.
2. Included Image Indices: Specify other images to include in the reconstruction.


## Components

### Main Script

- `main.py`:
This script handles:
1. Loading data from input files.
2. Parsing camera information and SIFT features.
3. Creating `ImageNode` objects for each input image.
4. Filtering nodes based on user input.
5. Matching image nodes using `ImageNodesMatch`.
6. Performing incremental registration with `PoseGraph`.
7. Saving the combined point cloud as `combined_point_cloud.ply`.

---

### Classes

- `ImageNode`:
Represents a single image in the reconstruction pipeline. 

- `ImageNodesMatch`:
Handles feature matching and transformation estimation between two images.

- `PoseGraph`:
Manages the global registration of multiple point clouds.