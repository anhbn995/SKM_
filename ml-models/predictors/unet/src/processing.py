from params import TMP_PATH
import cv2
import json
import uuid

import utils.ogr2ogr as ogr2ogr
from pathlib import Path
import glob

from utils.fs import mkdir
from unetprocessor import UnetRunning
from utils.vectorize import polygonize
from utils.image import get_quantile_schema
from utils.unet_data_preparation import *
from utils.fs import make_temp_folder


class PreprocessingUnet(UnetRunning):

    def __init__(self, bound_path, mask, data_dir, model_dir, image_path, unet_type, **kwargs):
        super().__init__(unet_type, **kwargs)
        self.bound_path = bound_path
        self.mask = mask
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.labels = kwargs.get("labels")
        self.input_type = kwargs.get("input_type")
        self.image_path = image_path
        with rasterio.open(self.image_path) as src:
            self.image_dtype = src.dtypes[0]

    def preprocess_run(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        masks_dir = '{}/masks'.format(self.model_dir)
        images_dir = '{}/images'.format(self.model_dir)

        datatrain_dir = [masks_dir, images_dir]
        for folder in datatrain_dir:
            if not os.path.exists(folder):
                os.makedirs(folder)

        with rasterio.open(self.image_path) as src:
            bands = src.read()
            profile = src.profile

        bands[np.isnan(bands)] = 0
        temp_folder = TMP_PATH

        new_image_path = f'{temp_folder}/image.tif'

        with rasterio.open(new_image_path, 'w', **profile) as src:
            src.write(bands)

        self.image_path = new_image_path
        qt_schema = self._get_quantile_schema()

        crop_raster_by_shape(self.image_path, self.bound_path, images_dir)

        self._validate_dataset(images_dir + '/*.tif')

        label = list(map(lambda el: el.get('value'), self.labels))

        ''' Generate mask with single band'''
        temp_mask_dir = None

        if self.input_type == 'raster':
            for (index, mask_el) in enumerate(self.mask):
                temp = np.asarray(mask_el).astype('uint8')
                num_file = len(fnmatch.filter(os.listdir(masks_dir), '*.tif'))
                temp_mask_dir = '{}/{}.tif'.format(masks_dir,
                                                   num_file + index + 1)
                h, w, crs, tr = get_info_mask(self.image_path, self.bound_path)
                arr2raster(temp_mask_dir, temp, 1, h, w, tr, crs)
        else:
            tmp_mask = '{}/mask.tif'.format(self.data_dir)
            h, w, crs, tr = get_info_image(self.image_path)
            arr2raster(tmp_mask, self.mask, 1, h, w, tr, crs)
            cropmask_from_area_shape(tmp_mask, self.bound_path, masks_dir)
            print(tmp_mask)
            # os.remove(tmp_mask)

        # Data generator
        train_dir = '{}/train'.format(self.data_dir)
        mkdir(paths=[train_dir])
        print("Generator...", images_dir, masks_dir)

        out_data_image_crop = main_gen_data_with_size(
            images_dir, masks_dir, train_dir, label, sampleSize=self.size)

        if temp_mask_dir:
            return temp_mask_dir

    def _validate_dataset(self, paths):
        error_message = ""
        has_valid_dataset = False
        for file in glob.glob(paths):
            dataset_image = rasterio.open(file)

            if dataset_image.width < self.size or dataset_image.height < self.size:
                error_message += 'Training area must be greater than {}x{}, current is {}x{}. '.format(
                    self.size, self.size, dataset_image.width, dataset_image.height)
                dataset_image = None
                os.remove(file)
                continue
            if np.max(np.array(dataset_image)) == 0:
                error_message += 'The area layer must be within the image. '
                dataset_image = None
                os.remove(file)
                continue
            has_valid_dataset = True
        if not has_valid_dataset:
            raise Exception(error_message)

    def _get_quantile_schema(self):
        if self.image_dtype == 'uint8':
            return None
        return get_quantile_schema(self.image_path)


class Postprocessing(UnetRunning):
    def __init__(self, input_tif, output_geojson, **kwargs):
        self.input_tif = input_tif
        self.output_geojson = output_geojson
        self.labels = kwargs.get('labels')

    def postprocess_run(self):

        with rasterio.open(self.input_tif, 'r', driver='GTiff') as src:
            img = src.read(1)
            crs = dict(src.crs)
            transform = src.transform
            w, h = src.width, src.height
            num_bands = src.count

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel3)
        for i in range(5):
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel2)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel2)
        for i in range(5):
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel2)

        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel3)

        mask_final = (1 * (image > 1)) * image

        mask_int = np.array(mask_final).astype('uint8')

        folder = Path(self.input_tif).parent
        str_id = uuid.uuid4().hex
        tmp_tif_path = '{}/{}.tif'.format(folder, str_id)
        tmp_geojson_path = '{}/{}.geojson'.format(folder, str_id)
        # mask = mask != 1
        result = rasterio.open(tmp_tif_path, 'w', driver='GTiff',
                               height=h, width=w,
                               count=num_bands, dtype='uint8',
                               crs=crs,
                               transform=transform,
                               compress='lzw')
        result.write(mask_int, 1)
        result.close()

        # list_pol= ["{}/gdal_polygonize.py".format(GDAL_BIN), "-of", "GeoJSON", "-b", "1", tmp_tif_path, tmp_geojson_path]
        # subprocess.call(list_pol)
        polygonize(tmp_tif_path, tmp_geojson_path)

        ogr2ogr.main(["", "-f", "geojson", '-t_srs', 'epsg:4326',
                     self.output_geojson, tmp_geojson_path])
        os.remove(tmp_tif_path)
        os.remove(tmp_geojson_path)
        with open(self.output_geojson) as outfile:
            data = json.load(outfile)

        with open(self.output_geojson, 'w') as outfile:
            data['labels'] = self.labels
            json.dump(data, outfile)

        return self.output_geojson

    def postprocess_run_farm(self):
        with rasterio.open(self.input_tif, 'r', driver='GTiff') as src:
            img = src.read(1)/255.0
            crs = dict(src.crs)
            affine = src.transform
            w, h = src.width, src.height

        # kernel = np.ones((5,5),np.uint8)
        print(1)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # affine.Affine(a, b, c,d, e, f) => Shapely (a, b, d, e, c, f)
        trans = np.array([affine[0], affine[1], affine[3],
                         affine[4], affine[2], affine[5]])
        # dilation
        img = cv2.dilate(img, kernel, iterations=1)
        # opening
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)

        # closing
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
        result = rasterio.open(self.output_geojson, 'w', driver='GTiff',
                               height=h, width=w,
                               count=1, dtype='uint8',
                               crs=crs,
                               transform=affine,
                               compress='lzw')
        result.write((img*2).astype(np.uint8), 1)
        result.close()
