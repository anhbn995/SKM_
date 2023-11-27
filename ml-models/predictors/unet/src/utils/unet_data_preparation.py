import os
import numpy as np
import geopandas as gp
import shapely
import random
import fiona
import rasterio
from tqdm import tqdm
import time
from rasterio.windows import Window
from rasterstats.io import Raster
import fnmatch
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly


def cropmask_from_area_shape(image_path, shape_path, result_dir):
    num_file = len(fnmatch.filter(os.listdir(result_dir), '*.tif'))
    path_result = result_dir + '/{}.tif'
    with rasterio.open(image_path) as src:
        crs = dict(src.crs)
        tr = src.transform
        w, h = src.width, src.height

    with Raster(image_path, band=1) as raster_obj_1:
        index = 1
        for feat in fiona.open(shape_path):
            polygon_geometry = feat['geometry']
            if polygon_geometry['type'] == 'Polygon':
                polygon = shapely.geometry.Polygon(
                    polygon_geometry['coordinates'][0])
            else:
                polygon = shapely.geometry.Polygon(
                    polygon_geometry['coordinates'][0][0])
            polygon_bounds = polygon.bounds
            raster_subset_1 = raster_obj_1.read(bounds=polygon_bounds)

            polygon_mask = rasterio.features.geometry_mask(geometries=[polygon_geometry],
                                                           out_shape=(
                                                               raster_subset_1.shape[0], raster_subset_1.shape[1]),
                                                           transform=raster_subset_1.affine,
                                                           all_touched=False,
                                                           invert=True)
            masked_1 = raster_subset_1.array * polygon_mask
            (h1, w1) = masked_1.shape
            result = rasterio.open(path_result.format(num_file + index), 'w', driver='GTiff',
                                   height=h1, width=w1,
                                   count=1, dtype="uint8",
                                   crs=crs,
                                   transform=raster_subset_1.affine,
                                   compress='lzw')
            result.write(masked_1, 1)
            result.close()
            index += 1


def crop_raster_by_shape(image_path, shape_path, result_dir):
    num_file = len(fnmatch.filter(os.listdir(result_dir), '*.tif'))
    path_result = result_dir + '/{}.tif'
    with rasterio.open(image_path, 'r', driver='GTiff') as src:
        crs = dict(src.crs)
        tr = src.transform
        w, h = src.width, src.height
        num_bands = src.count
        dtype = src.dtypes[0]

    index = 1
    for feat in fiona.open(shape_path):
        polygon_geometry = feat['geometry']
        if polygon_geometry['type'] == 'Polygon':
            polygon = shapely.geometry.Polygon(
                polygon_geometry['coordinates'][0])
        else:
            polygon = shapely.geometry.Polygon(
                polygon_geometry['coordinates'][0][0])

        polygon_bounds = polygon.bounds
        with Raster(image_path, band=1) as raster_obj:
            raster_subset = raster_obj.read(bounds=polygon_bounds)
            (h1, w1) = raster_subset.shape
            result = rasterio.open(path_result.format(num_file + index), 'w', driver='GTiff',
                                   height=h1, width=w1,
                                   count=num_bands, dtype=dtype,
                                   crs=crs,
                                   transform=raster_subset.affine,
                                   nodata=0,
                                   compress='lzw')

        for i in range(1, num_bands + 1):
            with Raster(image_path, band=i) as raster_obj:
                raster_subset = raster_obj.read(bounds=polygon_bounds)

                polygon_mask = rasterio.features.geometry_mask(geometries=[polygon_geometry],
                                                               out_shape=(raster_subset.shape[0],
                                                                          raster_subset.shape[1]),
                                                               transform=raster_subset.affine,
                                                               all_touched=False,
                                                               invert=True)
                masked = raster_subset.array * polygon_mask
                result.write(masked, i)

        result.close()
        index += 1


def arr2raster(path_out, bands, num_band, h, w, tr, crs, dtype='uint8', nodata=0):
    try:
        height, width = np.shape(bands)
    except:
        height = h
        width = w

    new_dataset = rasterio.open(path_out, 'w', driver='GTiff',
                                height=height, width=width,
                                count=num_band, dtype=dtype,
                                crs=crs,
                                transform=tr,
                                nodata=nodata,
                                compress='lzw')
    if num_band == 1:
        new_dataset.write(bands, 1)
    else:
        for i in range(num_band):
            new_dataset.write(bands[i], i + 1)
    new_dataset.close()


def shp2mask(path_shp, path_bound, attribute, values, h, w, tr, dtype='uint8'):
    print('Rasterize shp to tif')
    shp = gp.read_file(path_shp)
    bound = gp.read_file(path_bound)
    mask = np.zeros(shape=(h, w,), dtype=dtype)
    mask__ = rasterio.features.rasterize(bound['geometry'],
                                         out_shape=(h, w),
                                         transform=tr,
                                         dtype=dtype
                                         )
    mask = mask + 1 * mask__

    for i, attr__ in enumerate(values):
        qualified_layers = shp[shp[attribute] == attr__]

        if len(qualified_layers) == 0:
            continue
        mask__ = rasterio.features.rasterize(qualified_layers['geometry'],
                                             out_shape=(h, w),
                                             transform=tr,
                                             dtype=dtype
                                             )
        mask = mask + (i + 1) * mask__

    return mask


def generate(img_path, mask_path, size, label_count, train, qt_schema=None):
    ds = rasterio.open(img_path)
    bcount = ds.count
    w, h = ds.width, ds.height

    i_range = h - size
    j_range = w - size

    dsmask = rasterio.open(mask_path)
    random.seed(123)

    while (True):
        try:
            i = random.randrange(i_range)
            j = random.randrange(j_range)

            y = np.ones((size, size, label_count))
            window = Window(j, i, size, size)

            x = []

            for h in range(bcount):
                band_qt = qt_schema[h] if qt_schema else None
                band = ds.read(h + 1, window=window)
                if band_qt:
                    stretched_band = np.interp(
                        band, (band_qt.get('p2'), band_qt.get('p98')), (1, 255)).astype(float)
                else:
                    stretched_band = band
                nodatamask = (band == float(0))
                stretched_band[nodatamask] = 0
                x.append(stretched_band)
            x = np.transpose(np.array(x).astype(float) / 255.0, (1, 2, 0))

            bandmask = dsmask.read(1, window=window)
            for h in range(label_count):
                y[:, :, h] = (bandmask == h + 1).astype(float)
            # exist samples
            if np.max(y) != 0:
                yield (x, y)
            else:
                continue
        except Exception as e:
            continue


def get_info_mask(image_path, bound_crop_path):
    with rasterio.open(image_path) as src:
        crs = dict(src.crs)
    with Raster(image_path, band=1) as raster_obj_1:
        for feat in fiona.open(bound_crop_path):
            polygon_geometry = feat['geometry']
            if polygon_geometry['type'] == 'Polygon':
                polygon = shapely.geometry.Polygon(
                    polygon_geometry['coordinates'][0])
            else:
                polygon = shapely.geometry.Polygon(
                    polygon_geometry['coordinates'][0][0])
            polygon_bounds = polygon.bounds
            raster_subset_1 = raster_obj_1.read(bounds=polygon_bounds)
            (h, w) = raster_subset_1.array.shape
            transform = raster_subset_1.affine
    return h, w, crs, transform


def get_info_image(image_path):
    with rasterio.open(image_path) as src:
        crs = dict(src.crs)
        transform = src.transform
        w, h = src.width, src.height
    return h, w, crs, transform


class CD_GenerateTrainingDataset:
    def __init__(self, basefile, labelfile, sampleSize, labels, outputFolder, fileprefix):
        self.basefile = basefile
        # self.imagefile=imagefile
        self.labelfile = labelfile
        self.sampleSize = sampleSize
        self.outputFolder = outputFolder
        self.fileprefix = fileprefix
        self.labels = labels
        self.outputFolder_base = None
        self.outputFolder_image = None
        self.outputFolder_label = None

    def generateTrainingDataset(self, nSamples):
        self.outputFolder_base = os.path.join(self.outputFolder, "images")
        self.outputFolder_label = os.path.join(self.outputFolder, "masks")
        if not os.path.exists(self.outputFolder):
            os.makedirs(self.outputFolder)
        if not os.path.exists(self.outputFolder_base):
            os.makedirs(self.outputFolder_base)

        if not os.path.exists(self.outputFolder_label):
            os.makedirs(self.outputFolder_label)

        base = gdal.Open(self.basefile, GA_ReadOnly)
        # base = np.transpose(base, (1,2,0))
        raster = gdal.Open(self.labelfile, GA_ReadOnly)
        geo = raster.GetGeoTransform()
        proj = raster.GetProjectionRef()
        size_X = raster.RasterXSize
        size_Y = raster.RasterYSize

        icount = 0
        time_a = time.time()

        def gen_data(icount, label, listOfCoordinates):
            location_label = random.choice(listOfCoordinates)

            if location_label[0] > (size_X - self.sampleSize):
                if location_label[1] > (size_Y - self.sampleSize):
                    py = size_Y - self.sampleSize
                else:
                    py = int(location_label[1])
                px = size_X - self.sampleSize
            else:
                px = int(location_label[0])
                py = int(location_label[1])

            rband = raster.GetRasterBand(1).ReadAsArray(
                px, py, self.sampleSize, self.sampleSize)
            # print(time.time()-time_a)
            if np.count_nonzero(rband) > 0.005 * self.sampleSize * self.sampleSize:
                geo1 = list(geo)
                geo1[0] = geo[0] + geo[1] * px
                geo1[3] = geo[3] + geo[5] * py
                basefile_cr = os.path.join(self.outputFolder_base,
                                           self.fileprefix + '_{}_{:03d}.tif'.format(label, icount + 1))
                gdal.Translate(basefile_cr, base, srcWin=[
                               px, py, self.sampleSize, self.sampleSize])

                labelfile_cr = os.path.join(self.outputFolder_label,
                                            self.fileprefix + '_{}_{:03d}.tif'.format(label, icount + 1))
                gdal.Translate(labelfile_cr, raster, srcWin=[
                               px, py, self.sampleSize, self.sampleSize])
            # print(time.time()-time_a)

        for label in self.labels:
            icount = 0
            print('label value: ', label)
            data_arr = np.array(raster.GetRasterBand(1).ReadAsArray())
            label_map = np.where(data_arr == label)
            listOfCoordinates = list(zip(label_map[1], label_map[0]))

            if len(listOfCoordinates) != 0:
                with tqdm(total=round(nSamples // len(self.labels))) as pbar:
                    while icount < round(nSamples // len(self.labels)):
                        gen_data(icount, label, listOfCoordinates)
                        icount += 1
                        pbar.update()
            else:
                continue

        raster = None
        image = None
        base = None


def create_list_id(path):
    list_id = []
    for file in os.listdir(path):
        if file.endswith(".tif"):
            list_id.append(file[:-4])
    return list_id


def main_gen_data_with_size(base_path, mask_path, outputFolder, labels, sampleSize=512):
    list_id = create_list_id(base_path)
    for image_id in list_id:
        basefile = os.path.join(base_path, image_id + ".tif")
        # imagefile=os.path.join(image_path,image_id+".tif")
        labelfile = os.path.join(mask_path, image_id + ".tif")
        # sampleSize=1024
        if os.path.isfile(basefile) and os.path.isfile(labelfile):
            with rasterio.open(labelfile) as src:
                w, h = src.width, src.height
            numgen = w * h // ((sampleSize // 2) ** 2)
            if numgen >= 1000:
                numgen = 1000
            print('num gen : ', numgen)
            fileprefix = image_id

            gen = CD_GenerateTrainingDataset(
                basefile, labelfile, sampleSize, labels, outputFolder, fileprefix)
            gen.generateTrainingDataset(numgen)

    return outputFolder
