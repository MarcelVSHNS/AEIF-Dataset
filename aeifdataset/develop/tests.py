import aeifdataset as ad

import numpy as np
import open3d as o3d
from math import radians, sin, cos, sqrt, atan2
from build.lib.aeifdataset import DataRecord


def filter_points(points, x_range, y_range, z_range):
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    mask = (points['x'] < x_min) | (points['x'] > x_max) | \
           (points['y'] < y_min) | (points['y'] > y_max) | \
           (points['z'] < z_min) | (points['z'] > z_max)
    return points[mask]


def numpy_to_open3d(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Nur die x, y, z-Koordinaten
    return pcd


def structured_to_xyz(points):
    xyz = np.vstack((points['x'], points['y'], points['z'])).T  # Shape: (n, 3)
    return xyz


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, fpfh


def flat_distance(lat1, lon1, lat2, lon2):
    # Umrechnung von Grad zu Metern
    lat_dist = (lat2 - lat1) * 111320  # Breitengradabstand in Metern
    lon_dist = (lon2 - lon1) * 111320 * cos(radians(lat1))  # Längengradabstand in Metern, angepasst an die Breite

    # Euklidischer Abstand
    distance = sqrt(lat_dist ** 2 + lon_dist ** 2)
    return distance


if __name__ == '__main__':
    save_dir = '/mnt/cold_data/anonymisation/training/seq_8'
    dataset = ad.Dataloader("/mnt/hot_data/dataset/seq_8_dhbw")
    frame = DataRecord('/mnt/hot_data/dataset/seq_1_maille/id00999_2024-09-27_10-35-41.4mse')[14]

    ad.save_dataset_images_multithreaded(dataset, save_dir, use_raw=True, dtype='jpeg', num_cores=10)

    # Save one image as png or jpeg. Optional suffix can be applied.
    # ad.save_image(camera.image.image, output_path, f'{camera.image.get_timestamp()}_{camera_name}', dtype='jpeg')
    # ad.save_all_images_in_frame(frame, output_path, create_subdir=True)
    # ad.save_dataset_images_multithreaded(dataset, output_path, use_raw=True, num_cores=8)
    # Aktuelle Übereinstimmungen: 1831

    '''
    previous_transform = None
    fail_counter = 0
    match_counter = 0  # Zähler für erfolgreiche Übereinstimmungen

    for datarecord in dataset:
        for frame in datarecord:
            points_tower = frame.tower.lidars.UPPER_PLATFORM
            points_vehicle = frame.vehicle.lidars.TOP

            points_tower_xyz = structured_to_xyz(points_tower)
            points_vehicle_xyz = structured_to_xyz(points_vehicle)

            # Open3D-Punktwolken für beide Sensoren erstellen
            pcd_tower = numpy_to_open3d(points_tower_xyz)
            pcd_vehicle = numpy_to_open3d(points_vehicle_xyz)

            # Parameter anpassen
            voxel_size = 2.0  # Größerer Wert für Downsampling und Feature-Berechnung
            distance_threshold = voxel_size * 5  # Größere Schwelle für RANSAC-Registrierung

            # Wenn entweder `previous_transform` leer ist oder der Fehlschlagzähler 3 erreicht hat, führe RANSAC aus
            if previous_transform is None or fail_counter >= 3:
                # Vorverarbeitung der Punktwolken und Berechnung von FPFH-Features
                pcd_tower_down, fpfh_tower = preprocess_point_cloud(pcd_tower, voxel_size)
                pcd_vehicle_down, fpfh_vehicle = preprocess_point_cloud(pcd_vehicle, voxel_size)

                ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                    pcd_vehicle_down, pcd_tower_down, fpfh_vehicle, fpfh_tower,
                    False,
                    80.0,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    5,
                    [
                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.7),
                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(80.0)
                    ],
                    o3d.pipelines.registration.RANSACConvergenceCriteria(500000, 500)
                )

                # Transformationsmatrix nach RANSAC
                initial_transform = ransac_result.transformation
                fail_counter = 0  # Fehlschlagzähler zurücksetzen
            else:
                # Nutze die vorherige Transformation als Initialisierung für ICP
                initial_transform = previous_transform

            # Feinausrichtung mit ICP
            threshold = 1.0  # Maximale Distanz für Paarbildung in ICP
            icp_result = o3d.pipelines.registration.registration_icp(
                pcd_vehicle, pcd_tower, threshold, initial_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

            transformation_matrix = icp_result.transformation

            # Bewertung der ICP-Registrierung
            fitness = icp_result.fitness
            inlier_rmse = icp_result.inlier_rmse

            # Überprüfung der Fitness- und RMSE-Bedingung
            if 0.6 < fitness < 0.8 and inlier_rmse < 0.367:
                # Setze die Transformation als `previous_transform` für den nächsten Durchlauf
                previous_transform = transformation_matrix
                fail_counter = 0  # Fehlschlagzähler zurücksetzen
                match_counter += 1  # Zähler für erfolgreiche Übereinstimmung erhöhen
            else:
                # Falls die Bedingungen nicht erfüllt sind, erhöhe den Fehlschlagzähler
                fail_counter += 1
        print(f"Aktuelle Übereinstimmungen: {match_counter}")


    frame = dataset[0][0]
    for datarecord in dataset:
        for frame in datarecord:
            speed = np.linalg.norm(frame.vehicle.DYNAMICS.velocity[0].linear_velocity) * 3.6
            if speed < 1:
                print(f'Datarecord: {datarecord.name}, Frame: {frame.frame_id}')
    
    # image = frame.vehicle.cameras.STEREO_LEFT
    
    
    points_left = frame.vehicle.lidars.LEFT
    points_top = frame.vehicle.lidars.TOP
    points_right = frame.vehicle.lidars.RIGHT
    
        ad.show_points(
        (points_left, (255, 0, 0)),
        (points_top, (0, 255, 0)),
        (points_right, (0, 0, 255))
    )

    ad.show_points(
        (points_left, (255, 0, 0)),
        (points_top, (0, 255, 0))
    )
    
    xyz_points = np.stack(
        (points_left['x'], points_left['y'], points_left['z']), axis=-1)
    visualize_lidar_points(xyz_points, title='Upper Platform LiDAR Point Cloud')
    
    LEFT
    x_range = (-2.9, 1.8)
    y_range = (-1.7, 1.6)
    z_range = (-2.8, 0.2)

    RIGHT
    x_range = (-1.2, 1.5)
    y_range = (-0.6, 1.7)
    z_range = (-1.1, 0)
    
    new_pts = filter_points(points_right, x_range, y_range, z_range)
    coordinates = np.vstack((new_pts['x'], new_pts['y'], new_pts['z'])).T
    ad.show_points(points_right)
    '''
    # ad.save_image(image, '/mnt/hot_data/samples')
    # ad.show_points(points)

    # ad.show_tf_correction(image, points, -0.003, -0.01, -0.004)
    # ad.get_projection_img(image, points).show()
    # ad.get_projection_img(image2, points).show()
