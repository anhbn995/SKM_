'''
# preprocessing.py
# parameters:
    - imagefile:
'''
import os
import math
import json
from pathlib import Path
import fiona
import rasterio
import numpy as np

from osgeo import gdal
from rasterio import features as rio_features
from shapely.geometry import shape, mapping

from training import train


def _make_dirs_if_not_exist(*dir_paths):
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)


def _get_image_crs(image_path):
    datasource = rasterio.open(image_path)
    return datasource.crs


def _get_json_file_content(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def _latest_index_in_folder(path: Path, pattern: str = "*"):
    try:
        files = path.glob(pattern)
        latest_file = max(files, key=lambda x: x.stat().st_ctime)
        return int(latest_file.stem)
    except Exception:
        return 0


def _transform_polygon_to_line(file_path):
    data = _get_json_file_content(file_path)
    transformed_features = []
    for feature in data.get('features'):
        shapely_geometry = shape(feature.get('geometry'))
        transformed_features.append({
            'type': 'Feature',
            'geometry': mapping(shapely_geometry.boundary)
        })
    out_data = {
        'type': 'FeatureCollection',
        'features': transformed_features
    }
    with open(file_path, 'w') as editor:
        editor.write(json.dumps(out_data))


def _write_dict_to_file(data, file_path):
    with open(file_path, 'w') as file:
        file.write(data)
    return True


def _clip_image(image_path, out_path, bound_path):
    with rasterio.open(image_path) as datasource:
        nodata = datasource.nodata or 0
    gdal.Warp(
        out_path,
        image_path,
        cutlineDSName=bound_path,
        cropToCutline=True,
        dstNodata=nodata,
        dstSRS='EPSG:4326'
    )


def _generate_maskfile(imagefile, labelshapefile, maskfile):
    with fiona.open(labelshapefile, "r") as shapefile:
        labels = [feature["geometry"] for feature in shapefile]

    with rasterio.open(imagefile) as src:
        height = src.height
        width = src.width
        src_transform = src.transform
        out_meta = src.meta.copy()
        out_meta.update({'count': 1})
        nodatamask = src.read_masks(1) == 0

    mask = rio_features.geometry_mask(
        labels,
        (height, width),
        src_transform,
        all_touched=True,
        invert=True
    ).astype(np.uint8)
    mask[nodatamask] = 0

    with rasterio.open(maskfile, "w", **out_meta) as dest:
        dest.write(mask, indexes=1)


def _generate_training_dataset(imagefile, maskfile, folder, size=480, stride=200):
    image_folder = f'{folder}/images'
    mask_folder = f'{folder}/masks'
    image_pattern = os.path.join(folder, "images", "{}.npz")
    mask_pattern = os.path.join(folder, "masks", "{}.npz")
    count = _latest_index_in_folder(Path(f'{folder}/images'), '*.npz') + 1

    _make_dirs_if_not_exist(image_folder, mask_folder)

    with rasterio.open(imagefile) as img_source:
        with rasterio.open(maskfile) as mask_source:
            image = img_source.read().transpose(1, 2, 0)
            mask = mask_source.read(1)
            h, w = image.shape[:2]
            if h < size or w < size:  # padding if the width or the height is smaller the size
                bottom_pad = max(0, size-h)
                right_pad = max(0, size-w)

                # padding image and mask
                image = np.pad(
                    image,
                    ((0, bottom_pad), (0, right_pad), (0, 0)),
                    mode='constant',
                    constant_values=0
                )
                mask = np.pad(
                    mask,
                    ((0, bottom_pad), (0, right_pad), (0, 0)),
                    mode='constant',
                    constant_values=0
                )
                # update new h, w
                h, w = image.shape[:2]

            step_x = 1+math.ceil((w-size)/stride)
            step_y = 1+math.ceil((h-size)/stride)
            origin_x = [i*stride for i in range(step_x)]
            origin_x[-1] = w-size  # update last patch along x axis
            origin_y = [i*stride for i in range(step_y)]
            origin_y[-1] = h-size  # update last patch along y axis
            for y in origin_y:
                for x in origin_x:
                    patch = image[y:y+size, x:x+size, :]
                    patch_mask = mask[y:y+size, x:x+size]
                    if(np.count_nonzero(patch_mask) == 0):
                        continue
                    # check if there are labels
                    if np.max(patch_mask) > 0:
                        np.savez_compressed(
                            image_pattern.format(count), arr=patch)
                        np.savez_compressed(
                            mask_pattern.format(count), arr=patch_mask)
                    count += 1


def transform_image_annotation_2_dataset(image_path, bound_path, annotation_path, working_path, index=0):
    clipped_image_path = f'{working_path}/{index}.tif'
    masked_image_path = f'{working_path}/{index}_mask.tif'
    training_dataset_folder = f'{working_path}'

    _transform_polygon_to_line(annotation_path)
    _clip_image(image_path, clipped_image_path, bound_path)

    _generate_maskfile(clipped_image_path, annotation_path, masked_image_path)
    _generate_training_dataset(
        clipped_image_path, masked_image_path, training_dataset_folder)


if __name__ == '__main__':
    image_path = '/home/nghipham/Desktop/grid3.tif'
    annotation_path = '/home/nghipham/Desktop/grid_4326.geojson'
    bound_path = '/home/nghipham/Desktop/bound_4326.geojson'
    working_folder = '/home/nghipham/source/rq-worker/processors/models/edge_detection/temp'
    model_folder = '/home/nghipham/source/rq-worker/processors/models/edge_detection/model'
    annotation = _get_json_file_content(annotation_path)
    bound = _get_json_file_content(bound_path)
    transform_image_annotation_2_dataset(
        image_path, bound, annotation, working_folder)
    train(working_folder, model_folder)
