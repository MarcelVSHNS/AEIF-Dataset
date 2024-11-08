"""
This module provides functions for processing and handling images related to camera sensors.
It includes functionalities for image rectification, depth map computation, saving images with metadata,
loading images with embedded metadata, and saving all images from a frame.

Functions:
    get_rect_img(data, performance_mode=False): Rectify the provided image using the camera's intrinsic and extrinsic parameters.
    get_disparity_map(camera_left, camera_right, stereo_param=None): Compute a disparity map from a pair of stereo images.
    get_depth_map(camera_left, camera_right, stereo_param=None): Generate a depth map from a pair of stereo camera images.
    save_image(image, output_path, suffix='', metadata=None): Save an image to disk with optional metadata.
    save_all_images_in_frame(frame, output_path, create_subdir=False, use_raw=False): Save all images from a frame's vehicle and tower cameras.
    load_image_with_metadata(file_path): Load an image along with its embedded metadata.
"""
from typing import Optional, Tuple, Union
import os
from PIL import Image as PilImage
from PIL.PngImagePlugin import PngInfo
from io import BytesIO
from aeifdataset.data import CameraInformation, Camera, Image
import numpy as np
import cv2


def get_rect_img(data: Union[Camera, Tuple[Image, CameraInformation]], performance_mode: bool = False) -> Image:
    """Rectify the provided image using either a Camera object or an Image with CameraInformation.

    Performs image rectification using the camera matrix, distortion coefficients, rectification matrix,
    and projection matrix. The rectified image is returned as an `Image` object.

    Args:
        data (Union[Camera, Tuple[Image, CameraInformation]]): Either a Camera object containing the image and calibration parameters,
            or a tuple of an Image object and a CameraInformation object.
        performance_mode (bool, optional): If True, faster interpolation (linear) will be used; otherwise, higher quality (Lanczos4) will be used. Defaults to False.

    Returns:
        Image: The rectified image wrapped in the `Image` class.
    """
    if isinstance(data, Camera):
        # Handle the case where a Camera object is passed
        image = data._image_raw
        camera_info = data.info
    else:
        # Handle the case where an Image and CameraInformation are passed
        image, camera_info = data

    # Perform the rectification
    mapx, mapy = cv2.initUndistortRectifyMap(
        cameraMatrix=camera_info.camera_mtx,
        distCoeffs=camera_info.distortion_mtx[:-1],
        R=camera_info.rectification_mtx,
        newCameraMatrix=camera_info.projection_mtx,
        size=camera_info.shape,
        m1type=cv2.CV_16SC2
    )

    interpolation_algorithm = cv2.INTER_LINEAR if performance_mode else cv2.INTER_LANCZOS4

    # Convert image to numpy array and perform rectification
    rectified_image = cv2.remap(np.array(image.image), mapx, mapy, interpolation=interpolation_algorithm)

    # Return the rectified image wrapped in the Image class with its timestamp
    return Image(PilImage.fromarray(rectified_image), image.timestamp)


def get_disparity_map(camera_left: Camera, camera_right: Camera,
                      stereo_param: Optional[cv2.StereoSGBM] = None) -> np.ndarray:
    """Compute a disparity map from a pair of stereo images.

    This function computes a disparity map using stereo block matching.
    The disparity map is based on the rectified grayscale images of the stereo camera pair.

    Args:
        camera_left (Camera): The left camera of the stereo pair.
        camera_right (Camera): The right camera of the stereo pair.
        stereo_param (Optional[cv2.StereoSGBM]): Optional custom StereoSGBM parameters for disparity calculation.
                                                 If not provided, default parameters will be used.

    Returns:
        np.ndarray: The computed disparity map.
    """
    img1_gray = np.array(camera_left.image.convert('L'))
    img2_gray = np.array(camera_right.image.convert('L'))

    stereo = stereo_param or _create_default_stereo_sgbm()
    disparity_map = stereo.compute(img1_gray, img2_gray).astype(np.float32)

    return disparity_map


def _create_default_stereo_sgbm() -> cv2.StereoSGBM:
    """Create default StereoSGBM parameters for disparity computation."""
    window_size = 5
    min_disparity = 0
    num_disparities = 128  # Must be divisible by 16
    block_size = window_size

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,  # P1 and P2 control the smoothness
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    return stereo


def get_depth_map(camera_left: Camera, camera_right: Camera,
                  stereo_param: Optional[cv2.StereoSGBM] = None) -> np.ndarray:
    """Generate a depth map from a pair of stereo camera images (Experimental).

    This function computes the depth map by first calculating the disparity map between the left and right
    camera images, and then converting the disparity map to a depth map using the right camera's intrinsic parameters.

    Note: This function is experimental and has not been extensively tested on real-world data. The quality of the results may vary.

    Args:
        camera_left (Camera): The left camera of the stereo camera pair.
        camera_right (Camera): The right camera of the stereo camera pair. The intrinsic and extrinsic parameters
                               from this camera are used for disparity-to-depth conversion.
        stereo_param (Optional[cv2.StereoSGBM]): Optional StereoSGBM parameter object for controlling the stereo matching.
                                                 If not provided, default parameters will be used for disparity calculation.

    Returns:
        np.ndarray: The computed depth map.
    """
    disparity_map = get_disparity_map(camera_left, camera_right, stereo_param)

    depth_map = disparity_to_depth(disparity_map, camera_right)

    return depth_map


def disparity_to_depth(disparity_map: np.ndarray, camera_info: Union[Camera, CameraInformation]) -> np.ndarray:
    """Convert a disparity map to a depth map using camera parameters (Experimental).

    This function converts a disparity map into a depth map using the intrinsic parameters of the camera.

    Note: This function is experimental and has not been extensively tested on real-world data. The quality of the results may vary.

    Args:
        disparity_map (np.ndarray): The disparity map to convert to depth.
        camera_info (Union[Camera, CameraInformation]): The Camera object or CameraInformation object containing 
                                                   the focal length and baseline information.

    Returns:
        np.ndarray: The computed depth map, with masked areas where disparity is zero.
    """
    if hasattr(camera_info, 'info'):
        camera_info = camera_info.info
    else:
        camera_info = camera_info

    focal_length = camera_info.camera_mtx[0][0]
    baseline = abs(camera_info.stereo_transform.translation[0])

    # Calculate depth map, set depth to np.inf where disparity is zero
    with np.errstate(divide='ignore'):  # Ignore divide by zero warnings
        depth_map = np.where(disparity_map > 0, (focal_length * baseline) / disparity_map, np.inf)

    return depth_map


def save_image(image: Image, output_path: str, suffix: str = '', format: str = 'JPEG'):
    """Save an image to disk in JPEG or PNG format.

    Saves an `Image` object to disk in the specified format (JPEG or PNG).
    No additional compression is applied to JPEG images.

    Args:
        image (Image): The image to be saved.
        output_path (str): The directory where the image will be saved.
        suffix (str, optional): Optional suffix to be added to the image filename. Defaults to ''.
        format (str, optional): Format in which to save the image ('JPEG' or 'PNG'). Defaults to 'PNG'.
    """
    ext = 'jpg' if format.upper() == 'JPEG' else 'png'
    output_file = os.path.join(output_path, f'{image.get_timestamp()}{suffix}.{ext}')
    with open(output_file, 'wb') as file:
        file.write(image.image)


def save_all_images_in_frame(frame, output_path: str, create_subdir: bool = False, use_raw: bool = False):
    """Save all images from a frame's vehicle and tower cameras.

    Iterates through all cameras in the frame, saving each camera's image.
    If `create_subdir` is True, a subdirectory for each camera will be created.
    If `use_raw` is True, the raw image (`camera._image_raw`) will be saved, otherwise
    the processed image (`camera.image`) will be used.

    Args:
        frame: The frame object containing vehicle and tower cameras.
        output_path (str): The directory where images will be saved.
        create_subdir (bool, optional): If True, creates a subdirectory for each camera. Defaults to False.
        use_raw (bool, optional): If True, saves the raw image (`camera._image_raw`), otherwise saves the processed image.
                                  Defaults to False.
    """
    os.makedirs(output_path, exist_ok=True)
    for agent in frame:
        for camera_name, camera in agent.cameras:
            # Use raw image if 'use_raw' is True, otherwise use the processed image
            image_to_save = camera._image_raw if use_raw else camera.image

            if create_subdir:
                camera_dir = os.path.join(output_path, camera_name.lower())
                os.makedirs(camera_dir, exist_ok=True)
                save_path = camera_dir
                save_image(image_to_save, save_path, '')
            else:
                save_path = output_path
                save_image(image_to_save, save_path, f"_{camera_name.lower()}")


def load_image_with_metadata(file_path: str) -> Tuple[PilImage.Image, dict]:
    """Load an image along with its metadata.

    Loads an image file and extracts any embedded metadata.

    Args:
        file_path (str): The path to the image file.

    Returns:
        Tuple[PilImage.Image, dict]: The loaded image and a dictionary containing the metadata.
    """
    image = PilImage.open(file_path)

    metadata = image.info
    metadata_dict = {}
    for key, value in metadata.items():
        metadata_dict[key] = value.decode('utf-8') if isinstance(value, bytes) else value

    return image, metadata_dict
