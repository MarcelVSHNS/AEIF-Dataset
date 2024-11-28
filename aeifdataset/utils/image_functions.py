"""
This module provides functions for processing and handling images related to camera sensors and datasets.
It includes functionalities for image rectification, disparity and depth map computation, and saving images
with or without metadata. Additionally, it supports saving all images from frames or entire datasets using
multithreading for efficient processing.

Functions:
    get_rect_img(data, performance_mode=False): Rectify the provided image using the camera's intrinsic and extrinsic parameters.
    get_disparity_map(camera_left, camera_right, stereo_param=None): Compute a disparity map from a pair of stereo images.
    get_depth_map(camera_left, camera_right, stereo_param=None): Generate a depth map from a pair of stereo camera images.
    disparity_to_depth(disparity_map, camera_info): Convert a disparity map into a depth map using camera parameters.
    save_image(image, output_path, filename, dtype='PNG'): Save an image to disk in the specified format ('JPEG' or 'PNG').
    save_all_images_in_frame(frame, output_path, create_subdir=True, use_raw=False, dtype='PNG'): Save all images from a frame's cameras to the specified directory.
    _save_datarecord_images(datarecord, save_dir, create_subdir, use_raw, dtype): Save all images from frames in a datarecord.
    save_dataset_images_multithreaded(dataset, save_dir, create_subdir=True, use_raw=False, dtype='PNG', num_cores=2): Save all images from a dataset using multithreading for faster processing.
"""
from typing import Optional, Tuple, Union
import os
from PIL import Image as PilImage
from aeifdataset.data import CameraInformation, Camera, Image
import numpy as np
import cv2
import multiprocessing as mp
import sys


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


def save_image(image: Union['Image', PilImage.Image], output_path: str, filename: str, dtype: str = 'PNG'):
    """Saves a single image to disk in the specified format.

    This function saves an image (raw or processed) to a specified directory with a given filename.
    The supported formats are 'JPEG' and 'PNG'.

    Args:
        image (Union[Image, PilImage.Image]): The image to be saved. If an `Image` object is provided,
            its internal `PilImage` representation is used.
        output_path (str): The directory where the image will be saved.
        filename (str): The name of the file (without extension).
        dtype (str, optional): The format in which to save the image ('JPEG' or 'PNG'). Defaults to 'PNG'.

    Raises:
        ValueError: If an unsupported format is specified.
    """
    if isinstance(image, Image):
        image = image.image

    dtype = dtype.upper()
    if dtype == "JPEG":
        ext = "jpg"
    elif dtype == "PNG":
        ext = "png"
    else:
        raise ValueError("Unsupported format. Use 'JPEG' or 'PNG'.")

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f'{filename}.{ext}')
    with open(output_file, 'wb') as file:
        file.write(image)


def save_all_images_in_frame(frame, output_path: str, create_subdir: bool = True, use_raw: bool = False,
                             dtype: str = 'PNG'):
    """Saves all images from the cameras in a frame.

    This function iterates through all cameras in the given frame and saves their images
    to the specified output directory. It optionally creates subdirectories for each camera
    and supports saving raw or processed images in the specified format.

    Args:
        frame: The frame object containing vehicle and tower cameras.
        output_path (str): The directory where images will be saved.
        create_subdir (bool, optional): Whether to create subdirectories for each camera. Defaults to True.
        use_raw (bool, optional): Whether to save raw images instead of processed images. Defaults to False.
        dtype (str, optional): The format in which to save the images ('JPEG' or 'PNG'). Defaults to 'PNG'.

    Raises:
        ValueError: If an unsupported format is specified.
    """
    os.makedirs(output_path, exist_ok=True)
    for agent in frame:
        for camera_name, camera in agent.cameras:
            # Use raw image if 'use_raw' is True, otherwise use the processed image
            image_to_save = camera._image_raw if use_raw else camera.image
            timestamp = image_to_save.get_timestamp()

            if create_subdir:
                camera_dir = os.path.join(output_path, camera_name.lower())
                os.makedirs(camera_dir, exist_ok=True)
                save_path = camera_dir
                save_image(image_to_save, output_path=save_path, filename=timestamp, dtype=dtype)
            else:
                save_path = output_path
                save_image(image_to_save, output_path=save_path, filename=f'{timestamp}_{camera_name.lower()}',
                           dtype=dtype)


def _save_datarecord_images(datarecord, save_dir, create_subdir, use_raw, dtype):
    """Saves all images from the frames in a datarecord.

    This function iterates through all frames in the given datarecord and saves the images
    from each frame's cameras. It supports saving raw or processed images in the specified format.

    Args:
        datarecord: The datarecord containing frames to process.
        save_dir (str): The directory where images will be saved.
        create_subdir (bool): Whether to create subdirectories for cameras.
        use_raw (bool): Whether to save raw images instead of processed images.
        dtype (str): The format in which to save the images ('JPEG' or 'PNG').
    """
    for frame in datarecord:
        save_all_images_in_frame(frame, save_dir, create_subdir, use_raw, dtype)


def save_dataset_images_multithreaded(dataset, save_dir: str, create_subdir: bool = True, use_raw: bool = False,
                                      dtype='PNG', num_cores: int = 2):
    """Saves images from a dataset using multithreading.

    Args:
        dataset: Iterable containing datarecords to process.
        save_dir (str): Directory where images will be saved.
        create_subdir (bool, optional): Whether to create subdirectories for cameras. Defaults to True.
        use_raw (bool, optional): Whether to use raw images instead of processed images. Defaults to False.
        dtype (str, optional): The data type in which to save the image ('PNG' or 'JPEG'). Defaults to 'PNG'.
        num_cores (int, optional): Number of cores to use for multithreading. Defaults to 2.
    """

    with mp.Pool(processes=num_cores) as pool:
        batch = []
        total_records = len(dataset)

        for i, datarecord in enumerate(dataset, start=1):
            batch.append(datarecord)

            if len(batch) == num_cores:
                results = [
                    pool.apply_async(_save_datarecord_images, args=(record, save_dir, create_subdir, use_raw, dtype))
                    for record in batch]

                for result in results:
                    try:
                        result.wait()
                    except Exception as e:
                        print(f"Error in worker process: {e}")

                batch.clear()

            percent_complete = (i / total_records) * 100
            sys.stdout.write(f"\rDataset Progress: {percent_complete:.2f}%")
            sys.stdout.flush()

        if batch:
            results = [pool.apply_async(_save_datarecord_images, args=(record,)) for record in batch]
            for result in results:
                try:
                    result.wait()
                except Exception as e:
                    print(f"Error in worker process: {e}")

        sys.stdout.write("\rDataset Progress: 100.00%\n")
        sys.stdout.flush()
