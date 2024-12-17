from scipy.ndimage import map_coordinates
from PIL import Image
import numpy as np

def load_img(img_path):
    return np.array(Image.open(img_path))

def compute_homography(src_pts, dst_pts):
    """Compute the homography matrix using the Direct Linear Transformation (DLT) algorithm."""
    
    assert src_pts.shape[0] == dst_pts.shape[0], "Number of source and destination points must match."
    assert src_pts.shape[0] >= 4, "At least 4 points are required to compute a homography."
    
    num_points = src_pts.shape[0]
    A = []
    
    for i in range(num_points):
        x, y = src_pts[i]
        x_prime, y_prime = dst_pts[i]
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
    
    A = np.array(A)
    
    U, S, Vt = np.linalg.svd(A)

    # Last column of V (or row of Vt)
    h = Vt[-1]
    
    H = h.reshape(3, 3)

    # Normalizing to compare with OpenCV
    H = H / H[-1, -1] 
    return H

def warp_image(img, H, output_shape):
    """
    Applies a homography matrix H to warp an image.
    
    Parameters:
        img (ndarray): Input image (H x W x C).
        H (ndarray): Homography matrix (3x3).
        output_shape (tuple): Desired shape of the output (height, width).
        
    Returns:
        warped_img (ndarray): The warped image.
    """

    height, width = output_shape
    y, x = np.indices((height, width))
    coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x.ravel())], axis=0)

    H_inv = np.linalg.inv(H)
    transformed_coords = H_inv @ coords
    transformed_coords /= transformed_coords[2]
    
    x_src = transformed_coords[0].reshape(height, width)
    y_src = transformed_coords[1].reshape(height, width)

    warped_img = np.zeros((height, width, img.shape[2]), dtype=img.dtype)

    for i in range(img.shape[2]):  # For each color channel
        warped_img[:, :, i] = map_coordinates(img[:, :, i], [y_src, x_src], order=1, mode='constant', cval=0)

    return warped_img


def transform_points(points, H):
    """
    Transforms a set of 2D points using a given homography matrix.
    Parameters:
    points (numpy.ndarray): An array of shape (N, 2) representing N 2D points.
    H (numpy.ndarray): A 3x3 homography matrix.
    Returns:
    numpy.ndarray: An array of shape (N, 2) representing the transformed 2D points.
    """

    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    
    transformed_points_homogeneous = np.dot(H, points_homogeneous.T).T
    
    transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2:3]
    
    return transformed_points