import rasterio
import math
import tensorflow as tf

from utils.component_reader import DaReader, DaWriterKernel, DaStretchKernel, DaUnetPredictKernel, \
    DaSyntheticKernel
from utils.image import get_quantile_schema
import numpy as np


class BasePredictor():
    def __init__(self, image_path, model_path, output_path, **kwargs):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        input_shape = self.model.input.shape.as_list()
        if type(self.model.output) is list:
            self.number_output = len(self.model.output)
            out_shape = self.model.output[0].shape.as_list()
        else:
            out_shape = self.model.output.shape.as_list()
            self.number_output = 1

        self.numbands_predict = int(input_shape[3])
        self.size = int(input_shape[2])
        self.output_class = int(out_shape[3])

        self.image_path = image_path
        self.output = output_path
        self.on_processing = kwargs.get('on_processing')
        self.labels = []
        for i in range(self.output_class):
            self.labels.append({"value": i + 2, "name": f"Label {i + 1}", "color": random_color()})

    def predict(self, ids_output=None):
        if not ids_output:
            ids_output = list(range(self.number_output))
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
            profile['nodata'] = 0
            profile['count'] = 1
            profile['dtype'] = 'uint8'

            size = self.size
            padding = size // 16
            size_predict = size - size // 8
            list_value = list(map(lambda el: el.get('value'), self.labels))

            stretch_kernel = DaStretchKernel(quantile_schema, profile)
            predictor_kernel = DaUnetPredictKernel(self.model, 'unet_farm', size, padding)

            results = [None] * self.number_output
            for i in ids_output:
                result_path = f'{self.output}/result_{i}.tif'
                results[i] = result_path

            writer_kernel = DaWriterKernel(results, list_value, ids_output, **profile)

            synthetic = DaSyntheticKernel([stretch_kernel, predictor_kernel, writer_kernel])
            img.multi_execute([synthetic], notify=notify, size=size_predict, buffer=padding)


def random_color():
    x = np.random.random(size=3) * 256
    r, g, b = tuple(list(x.astype(int)))
    return "#{:02x}{:02x}{:02x}".format(r, g, b)
