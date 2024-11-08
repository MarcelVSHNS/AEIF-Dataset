import multiprocessing as mp
import aeifdataset as ad
from aeifdataset import DataRecord
from aeifdataset.develop import show_tf_correction
from tqdm import tqdm
import numpy as np


def save_datarecord_images(datarecord, save_dir):
    for frame in datarecord:
        ad.save_all_images_in_frame(frame, save_dir, create_subdir=True, use_raw=True)


def save_dataset_images_multithreaded(dataset, save_dir, batch_size=10):
    num_workers = 6

    # Create the pool
    with mp.Pool(processes=num_workers) as pool:
        batch = []

        for i, datarecord in enumerate(tqdm(dataset, desc="Processing datarecords")):
            batch.append(datarecord)

            # Process the batch if it reaches the specified batch size
            if len(batch) == batch_size:
                results = [pool.apply_async(save_datarecord_images, args=(record, save_dir)) for record in batch]

                # Wait for the batch to complete before loading more data
                [result.wait() for result in results]

                # Clear the batch
                batch.clear()

        # Process any remaining datarecords in the batch
        if batch:
            results = [pool.apply_async(save_datarecord_images, args=(record, save_dir)) for record in batch]
            [result.wait() for result in results]

        pool.close()
        pool.join()


def filter_points(points, x_range, y_range, z_range):
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    mask = (points['x'] < x_min) | (points['x'] > x_max) | \
           (points['y'] < y_min) | (points['y'] > y_max) | \
           (points['z'] < z_min) | (points['z'] > z_max)
    return points[mask]


if __name__ == '__main__':
    save_dir = '/mnt/cold_data/anonymisation/training/seq_5'
    dataset = ad.Dataloader("/mnt/hot_data/dataset/seq_5_bahnhof")

    # save_dataset_images_multithreaded(dataset, save_dir)
    frame = dataset[3][0]

    image = frame.tower.cameras.VIEW_1
    points = frame.tower.lidars.UPPER_PLATFORM

    # image2 = frame.tower.cameras.VIEW_2

    # ad.save_image(image, '/mnt/hot_data/samples')
    # ad.show_points(points)

    # ad.show_tf_correction(image, points, -0.003, -0.01, -0.004)
    # ad.get_projection_img(image, points).show()
    # ad.get_projection_img(image2, points).show()
