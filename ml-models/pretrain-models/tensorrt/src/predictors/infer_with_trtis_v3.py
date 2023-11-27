import numpy as np
from osgeo import gdal
import cv2
import osr
from ml_lib.convert_datatype import list_contour_to_list_polygon, contour_to_polygon
from ml_lib.export_data import list_polygon_to_list_geopolygon, polygon_to_geopolygon, transformToLatLong
from shapely.geometry import Polygon
import fiona
from fiona.crs import from_epsg
from shapely import geometry
import os
import math
import geopandas as gpd
import rasterio
from utils.image import stretch_image
from tensorrtserver.api import ProtocolType, ServerHealthContext, ServerStatusContext, InferContext
from ml_lib.export_data import exportResult3 as exportResult3
from params import TENSORRT_SERVER_URL, GDAL_BIN
import uuid

gdal_merge = "{}/gdal_merge.py".format(GDAL_BIN)


class DeforestationPredict:
    def __init__(
            self,
            input,
            output,
            infer_ctx,
            input_tensor_name,
            output_tensor_name,
    ):
        self.infer_ctx = infer_ctx
        self.input = input
        self.output = output
        self.input_tensor_name = input_tensor_name
        self.output_tensor_name = output_tensor_name

    def predict(self):
        print('Predicting ...')
        size = 128
        strike = 96
        pad = 16
        # open dataset
        ds = gdal.Open(self.input, gdal.GA_ReadOnly)
        bcount = ds.RasterCount
        cols = ds.RasterXSize
        rows = ds.RasterYSize

        predictdata = np.zeros((rows, cols), dtype=np.uint8)
        image = np.zeros((rows, cols, 4), dtype=np.uint8)
        for i in range(bcount):
            band = np.array(ds.GetRasterBand(i + 1).ReadAsArray())
            image[..., i] = band

        rowsteps = int(math.ceil(rows / strike))
        colsteps = int(math.ceil(cols / strike))

        print("rows:{}, cols:{}, rowsteps:{}, colsteps:{}".format(
            rows, cols, rowsteps, colsteps))

        for i in range(rowsteps):
            X = np.zeros((colsteps, size, size, 4), dtype=np.float32)
            for j in range(colsteps):
                xrowstart = 0
                xrowend = size
                irowstart = i * strike - pad
                irowend = (i + 1) * strike + pad
                xcolstart = 0
                xcolend = size
                icolstart = j * strike - pad
                icolend = (j + 1) * strike + pad

                if i == 0:
                    xrowstart = pad
                    irowstart = i * strike
                if j == 0:
                    xcolstart = pad
                    icolstart = j * strike
                if i == rowsteps - 1:
                    xrowend = rows - i * strike + pad
                    irowend = rows
                if j == colsteps - 1:
                    xcolend = cols - j * strike + pad
                    icolend = cols
                if irowend > rows:
                    xrowend = xrowend - (irowend - rows)
                    irowend = rows
                if icolend > cols:
                    xcolend = xcolend - (icolend - cols)
                    icolend = cols

                X[j, xrowstart:xrowend, xcolstart:xcolend,
                    :] = image[irowstart:irowend, icolstart:icolend, :] / 255
            preds = []
            for tile in X:
                trtis_result = self.infer_ctx.run(
                    {
                        self.input_tensor_name: (tile,)
                    },
                    {
                        self.output_tensor_name: InferContext.ResultFormat.RAW
                    },
                    1  # Batch size
                )

                preds.append(trtis_result[self.output_tensor_name][0])
            preds = np.array(preds)
            preds_t = (preds > 0.4).astype(np.uint8)
            for j in range(colsteps):
                xrowstart = 0
                xrowend = size
                irowstart = i * strike - pad
                irowend = (i + 1) * strike + pad
                xcolstart = 0
                xcolend = size
                icolstart = j * strike - pad
                icolend = (j + 1) * strike + pad

                if i == 0:
                    xrowstart = pad
                    irowstart = i * strike
                if j == 0:
                    xcolstart = pad
                    icolstart = j * strike
                if i == rowsteps - 1:
                    xrowend = rows - i * strike + pad
                    irowend = rows
                if j == colsteps - 1:
                    xcolend = cols - j * strike + pad
                    icolend = cols

                # irowend>rows or icolend>cols
                if irowend > rows:
                    xrowend = xrowend - (irowend - rows)
                    irowend = rows
                if icolend > cols:
                    xcolend = xcolend - (icolend - cols)
                    icolend = cols

                predictdata[irowstart:irowend, icolstart:icolend] = preds_t[j, xrowstart:xrowend, xcolstart:xcolend,
                                                                            0] * 255

        with rasterio.open(self.input, 'r', driver='GTiff') as src:
            crs = dict(src.crs)
            transform = src.transform
        predictdata = 255 - predictdata  # deforestation - 0, background 255
        result = rasterio.open(self.output, 'w', driver='GTiff',
                               height=rows, width=cols,
                               count=1, dtype='uint8',
                               crs=crs,
                               transform=transform,
                               compress='lzw')
        result.write(predictdata, 1)

        result.close()

        ds = None


def infer(
        input,
        output,
        on_processing=None,
        url=TENSORRT_SERVER_URL,
        protocol='http',
        http_headers=None,
        model_name='deforest',
        model_version=None,
        async_set=False
):
    """
    Parameters:
        image_path: Path to input image.
        output_path: Path to where result is saved.
        url: Server's url.
        protocol: Protocol ("http"/"grpc") used to communicate with inference service. Default is "http".
        http_headers: HTTP headers to add to inference server requests. Format is -H"Header:Value".
        model_name: The name of selected model.
        model_version: The version of the model to use for inference,
            or None to indicate that the latest (i.e. highest version number) version should be used.
        async_set: Use asynchronous inference API.
        driver: Output's driver, e.i. GeoJSON, ESRI Shapefile, ...; default is GeoJSON.

    Returns:
        None
    """

    protocol = ProtocolType.from_str(protocol)

    # Create a health context, get the ready and live state of server.
    health_ctx = ServerHealthContext(url, protocol,
                                     http_headers=http_headers, verbose=False)
    print("Health for model {}".format(model_name))
    print("Live: {}".format(health_ctx.is_live()))
    print("Ready: {}".format(health_ctx.is_ready()))

    # Create a status context and get server status
    status_ctx = ServerStatusContext(url, protocol, model_name,
                                     http_headers=http_headers, verbose=False)
    print("Status for model {}".format(model_name))
    print(status_ctx.get_server_status())

    # Create the inference context for the model.
    infer_ctx = InferContext(url, protocol, model_name, model_version,
                             http_headers=http_headers, verbose=False)

    output_tiff = output.replace('.geojson', '.tif')
    str_id = uuid.uuid4().hex

    stretched_img_path = output.replace('.geojson', '{}.tif'.format(str_id))
    stretch_image(input, stretched_img_path)

    if model_name == 'deforest':
        DeforestationPredict(
            stretched_img_path,
            output_tiff,
            infer_ctx,
            input_tensor_name='img_2',
            output_tensor_name='conv2d_76/Sigmoid'
        ).predict()
    else:
        DeforestationPredict(
            stretched_img_path,
            output_tiff,
            infer_ctx,
            input_tensor_name='img',
            output_tensor_name='conv2d_19/Sigmoid'
        ).predict()

    # os.remove(stretched_img_path)
