import numpy as np
from PIL import Image as PilImage
from ameisedataset.data import Camera, Image
import cv2


def get_rect_img(camera: Camera):
    """Rectify the provided image using camera information."""
    # Init and calculate rectification matrix
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix=camera.info.camera_mtx,
                                             distCoeffs=camera.info.distortion_mtx[:-1],
                                             R=camera.info.rectification_mtx,
                                             newCameraMatrix=camera.info.projection_mtx,
                                             size=camera.info.shape,
                                             m1type=cv2.CV_16SC2)
    # Apply matrix
    rectified_image = cv2.remap(np.array(camera.image.image), mapx, mapy, interpolation=cv2.INTER_LANCZOS4)

    return Image(PilImage.fromarray(rectified_image), camera.image.timestamp)


def get_depth_map(camera_left: Camera, camera_right: Camera) -> np.ndarray:
    rect_left = get_rect_img(camera_left)
    rect_right = get_rect_img(camera_right)

    img1 = np.array(rect_left.image.convert('L'))  # Convert to grayscale
    img2 = np.array(rect_right.image.convert('L'))  # Convert to grayscale
    # Create the block matching algorithm with high-quality settings
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,  # Depending on the camera setup, this might need to be increased.
        blockSize=5,  # Smaller block size can detect finer details.
        P1=8 * 3 * 5 ** 2,  # Control smoothness of the disparity. Adjust as needed.
        P2=32 * 3 * 5 ** 2,  # Control smoothness. This is usually larger than P1.
        disp12MaxDiff=1,  # Controls maximum allowed difference in disparity check.
        uniquenessRatio=15,  # Controls uniqueness. Higher can mean more robustness against noise.
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # Utilizes 3-way dynamic programming. May provide more robust results.
    )
    # Compute the disparity map
    disparity = stereo.compute(img1, img2)
    # Normalize for better visualization
    disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # To avoid division by zero, set disparity values of 0 to a small value
    safe_disparity = np.where(disparity == 0, 0.000001, disparity)
    f = camera_right.info.focal_length
    b = abs(camera_right.info.stereo_transform.translation[0]) * 10 ** 3
    depth_map = f * b / safe_disparity
    return depth_map