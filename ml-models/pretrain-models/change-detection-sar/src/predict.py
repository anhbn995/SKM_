import rasterio
import math
import tensorflow as tf
import os
import pandas as pd
import csv
import time
from component_reader import DaReader, DaWriterKernel, \
    DaStretchKernel, DaUnetPredictKernel, DaSyntheticKernel, MorphologyKernel
from rio_tiler.io import COGReader
from datetime import date
import model_defores as model_lib
import glob
from tqdm import tqdm
from rasterio.windows import Window
from matplotlib.patches import Polygon
import rasterio.features
import fiona
from shapely.geometry import Polygon, mapping
from rasterio.crs import CRS
import numpy as np
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
from keras.backend import set_session


def get_quantile_schema(img):
    qt_scheme = []
    # with COGReader(img) as cog:
    #     stats = cog.stats()
    #     for _, value in stats.items():
    #         qt_scheme.append({
    #             'p2': value['percentiles'][0],
    #             'p98': value['percentiles'][1],
    #         })
    with rasterio.open(img) as r:
        num_band = r.count
        for chanel in range(1, num_band+1):
            data = r.read(chanel).astype(np.float16)
            data[data == 0] = np.nan
            qt_scheme.append({
                'p2': np.nanpercentile(data, 2),
                'p98': np.nanpercentile(data, 98),
            })
    print(qt_scheme)
    return qt_scheme


def predict_farm(image_path, output, model1, model2=None, model3=None, pbars=None):
    quantile_schema = get_quantile_schema(image_path)

    with DaReader(image_path) as img:
        profile = img.profile
        profile['compress'] = 'DEFLATE'
        profile['nodata'] = 0
        profile['count'] = 1

        size = 128
        padding = size//8
        size_predict = size - size//4
        stretch_kernel = DaStretchKernel(quantile_schema, profile)
        predictor_kernel = DaUnetPredictKernel(
            model1, model2, model3, size, padding)
        morphology_kernel = MorphologyKernel()
        writer_kernel = DaWriterKernel(output, **profile)
        synthetic = DaSyntheticKernel(
            [stretch_kernel, predictor_kernel, morphology_kernel, writer_kernel])
        img.multi_execute([synthetic], size=size_predict,
                          buffer=padding, pbars=pbars)


def main(input_path, output_path, model_path):
    # config = tf.compat.v1.ConfigProto()
    # # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # config.gpu_options.allow_growth=True
    # set_session(tf.compat.v1.Session(config=config))
    #
    #
    unet = model_lib.unet_basic(4, 128)
    unet.load_weights(model_path)
    # input_path = "/media/skymap/Data/model_to_eof/stack_feb_oct/stack_Feb_Oct.tif"
    # output_path = "/media/skymap/Data/model_to_eof/stack_feb_oct/stack_Feb_Oct_test_rs.tif"
    predict_farm(input_path, output_path, unet)


if __name__ == '__main__':
    # config = tf.compat.v1.ConfigProto()
    # # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # config.gpu_options.allow_growth=True
    # set_session(tf.compat.v1.Session(config=config))

    unet = model_lib.unet_basic(4, 64)
    unet.load_weights("./model_sar.h5")
    input_path = "/media/skymap/Data/model_to_eof/stack_feb_oct/stack_Feb_Oct.tif"
    output_path = "/media/skymap/Data/model_to_eof/stack_feb_oct/stack_Feb_Oct_test_rs.tif"
    predict_farm(input_path, output_path, unet)
