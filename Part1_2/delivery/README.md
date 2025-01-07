# README for Project Part 1.2

## File Descriptions

### Python Scripts
1. `main.py`
   - Entry point for the project.
   - Loads input data and processes it to compute homographies and transform YOLO detections.
   - Outputs transformed YOLO data and homography matrices for each video.

2. `homography.py`
   - Contains functions for computing homographies using RANSAC.
   - Implements methods to validate and propagate homographies across frames.

3. `ImageNode.py`
   - Defines the `ImageNode` class for encapsulating image data, keypoints, descriptors, and transformations.

4. `load_data.py`
   - Provides functions for loading images, keypoints, descriptors, and YOLO detections from input directories.
   - Supports structured handling of data across multiple directories.

5. `matches.py`
   - Implements descriptor matching between frames using Euclidean distance with cross-check validation.

8. `save_results.py`
   - Manages the saving of homography matrices and transformed YOLO detections in the required format.

9. `thresholds.py`
   - Provides methods to calculate thresholds for descriptor distances and minimum inliers.

---

## Usage Instructions

### Prerequisites
- Python 3.x installed with necessary libraries (`numpy`, `scipy`,  `tqdm`).

### Command-Line Usage
To run the project, use the following command:

```bash
python main.py ref_dir input1_dir output1_dir input2_dir output2_dir ... inputn_dir outputn_dir
```

#### Arguments:
- `ref_dir`: Directory containing the reference image (`img_ref.jpg`) and its keypoints/descriptors (`kp_ref.mat`).
- `inputK_dir`: Directory containing input image sequence and corresponding YOLO/keypoints files.
- `outputK_dir`: Directory where outputs for `inputK_dir` will be saved.

### Outputs
Each `outputK_dir` will contain:
1. `homographies.mat`
   - Stores the homography matrices for all frames to the reference frame.
2. `yolooutput_XXXX.mat`
   - Transformed YOLO detections for each frame.

## Assumptions

- The algorithm is limited for videos with continuous motion and directly linked videos. 
- YOLO detections may be absent for some frames; these are handled gracefully.
