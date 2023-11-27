import numpy as np
import uuid
import rasterio
from osgeo import osr, gdal
import os
from utils.image import get_quantile_schema
from utils.export_data import get_bound
from utils.export_data import export_predict_result_to_file
from mrcnn_processor import MrcnnRunning
from params import TMP_PATH
from preprocessing.gen_data_train import gen_trainning_dataset
from shutil import copyfile


def make_data_folder(tmp_folder, image, anno, bound):
    name = uuid.uuid4().hex
    image_dir = f'{tmp_folder}/images'
    anno_dir = f'{tmp_folder}/annotations'
    bound_dir = f'{tmp_folder}/bounds'
    for dir in [image_dir, anno_dir, bound_dir]:
        os.makedirs(dir)

    copyfile(image, f'{image_dir}/{name}.tif')
    if os.path.isfile(f'{image_dir}/{name}.tif'):
        print('copy successfulllllllll')
    copyfile(anno, f'{anno_dir}/{name}.geojson')
    copyfile(bound, f'{bound_dir}/{name}.geojson')
    return image_dir, anno_dir, bound_dir


class PreprocessingMrcnn(MrcnnRunning):
    def __init__(self, image_path, bound_path, annotation_path, data_dir, **kwargs):

        self.bound_path = bound_path
        self.annotation_path = annotation_path
        self.data_dir = data_dir
        self.image_path = image_path
        self.trainer_size = kwargs.get("trainer_size") or 512
        self.stride_size = kwargs.get("stride_size") or 256
        self.split_ratio = kwargs.get("split_ratio") or 0.8
        with rasterio.open(self.image_path) as src:
            self.image_dtype = src.dtypes[0]

    def preproces_run(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        with rasterio.open(self.image_path) as src:
            bands = src.read()
            profile = src.profile

        bands[np.isnan(bands)] = 0
        temp_folder = TMP_PATH

        img_id_hex = uuid.uuid4().hex
        new_image_path = f'{temp_folder}/{img_id_hex}.tif'

        with rasterio.open(new_image_path, 'w', **profile) as src:
            src.write(bands)
        self.image_path = new_image_path
        image_dir, anno_dir, bound_dir = make_data_folder(temp_folder, new_image_path, self.annotation_path,
                                                          self.bound_path)
        trainning_data_dir = gen_trainning_dataset(
            image_dir, bound_dir, anno_dir, self.data_dir)
        return trainning_data_dir

    def _get_quantile_schema(self):
        if self.image_dtype == 'uint8':
            return None
        return get_quantile_schema(self.image_path)


class PostprocessingMrcnn(MrcnnRunning):
    def __init__(self, list_polygons, list_score, output_path, image_path):
        self.list_polygons = list_polygons
        self.output_path = output_path
        self.image_path = image_path
        self.list_score = list_score

    def postprocess_run(self):
        dataset_image = gdal.Open(self.image_path)
        driverName = "GeoJSON"

        geotransform = dataset_image.GetGeoTransform()
        projection = osr.SpatialReference(dataset_image.GetProjectionRef())

        # polygons_result = list_contour_to_list_polygon(self.list_contours)
        # keep1 = nms_polygon_cpu(polygons_result,self.list_score)

        with rasterio.open(self.image_path) as dataset_image:
            # Read image information
            transform = dataset_image.transform
            w, h = dataset_image.width, dataset_image.height
            proj_str = (dataset_image.crs.to_string())
            # Get id image
            image_name = os.path.basename(self.image_path)
            # Get AOI by image ID
            bound_aoi = get_bound(self.image_path)
            # on_processing(0.95)
            # polygon_result_all = [polygons_result[i] for i in keep1]
            # exportResult2(polygons_result, geotransform, projection, self.output_path, driverName)
            # on_processing(0.99)
            export_predict_result_to_file(self.list_polygons, self.list_score, bound_aoi, transform, proj_str,
                                          self.output_path)

        return self.output_path
