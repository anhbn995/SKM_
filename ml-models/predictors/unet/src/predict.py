import rasterio
import math
import tensorflow as tf
import numpy as np
from component_reader import DaReader, DaWriterKernel, DaStretchKernel, DaUnetPredictKernel, DaSyntheticKernel, MorphologyKernel
from utils.image import get_quantile_schema

from unetprocessor import UnetRunning


class UnetPredictor(UnetRunning):
    def __init__(self, image_path, model_path, output_path, unet_type, **kwargs):
        super().__init__(unet_type, **kwargs)
        self.list_value = []
        for obj in kwargs.get('labels'):
            self.list_value.append(obj.get('value'))
        self.load_full_model = kwargs.get('load_full_model')
        self.image_path = image_path
        self.output = output_path
        self.numbands_predict = kwargs.get('numbands_predict')
        if self.numbands_predict:
            self.numbands = self.numbands_predict
        self.loadmodel(model_path)
        self.on_processing = kwargs.get('on_processing')

    def loadmodel(self, model_path):
        if self.load_full_model:
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = super().identify_model()
            self.model.load_weights(model_path)
        self.model.summary()

    def predict(self):
        with rasterio.open(self.image_path) as imggg:
            dtype = imggg.dtypes[0]
        if dtype == 'uint8':
            quantile_schema = None
        else:
            quantile_schema = get_quantile_schema(self.image_path)

        def notify(total, current):
            if self.on_processing:
                ammount_percent = math.ceil(total / 100)
                if current % ammount_percent < ammount_percent / 100:
                    self.on_processing(current / total)

        idx = None
        if self.numbands_predict:
            idx = tuple([n + 1 for n in range(self.numbands_predict)])

        with DaReader(self.image_path) as img:
            profile = img.profile
            profile['compress'] = 'DEFLATE'
            profile['dtype'] = 'uint8'
            profile['nodata'] = 0
            profile['count'] = 1
            stretch_kernel = DaStretchKernel(quantile_schema)

            print('quantile_schema', quantile_schema)
            print('stretch_kernel', stretch_kernel)

            padding = self.size // 16
            size_predict = self.size - self.size // 8
            predictor_kernel = DaUnetPredictKernel(
                self.model, self.unet_type, self.size, padding)
            writer_kernel = DaWriterKernel(
                [self.output], list_value=np.sort(self.list_value), **profile)
            synthetic = DaSyntheticKernel(
                [stretch_kernel, predictor_kernel, writer_kernel])
            img.multi_execute([synthetic], notify=notify,
                              size=size_predict, buffer=padding, idx=idx)

    def predict_farm(self):
        quantile_schema = get_quantile_schema(self.image_path)

        def notify(total, current):
            print(current / total)

        with DaReader(self.image_path) as img:
            profile = img.profile
            profile['compress'] = 'DEFLATE'
            profile['dtype'] = 'uint8'
            profile['nodata'] = 0
            profile['count'] = 1
            size = 480
            padding = size // 16
            size_predict = size - size // 8
            stretch_kernel = DaStretchKernel(quantile_schema)
            predictor_kernel = DaUnetPredictKernel(
                self.model, self.unet_type, size, padding)
            morphology_kernel = MorphologyKernel()
            writer_kernel = DaWriterKernel(self.output, **profile)
            synthetic = DaSyntheticKernel(
                [stretch_kernel, predictor_kernel, morphology_kernel, writer_kernel])
            img.multi_execute([synthetic], notify=notify,
                              size=size_predict, buffer=padding)
