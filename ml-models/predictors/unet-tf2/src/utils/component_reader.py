import cv2
import rasterio
import numpy as np

from rasterio.windows import Window
from rasterio.features import bounds as feature_bounds
from rio_tiler.io import COGReader
from rio_tiler.utils import non_alpha_indexes, create_cutline


class DaKernel:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, data, mask, window, shrink_window=None):
        pass


class DaReader:
    def __init__(self, assets):
        if isinstance(assets, str):
            self.assets = [assets]
        else:
            self.assets = assets
        if len(assets) == 0:
            raise Exception('Assets is empty')
        self._imageReaders = []
        for asset in self.assets:
            self._imageReaders.append(COGReader(asset))
        self.closed = False

    @property
    def imageReaders(self):
        return self._imageReaders

    @property
    def band_count(self):
        b_count = 0
        for img in self.imageReaders:
            idx = non_alpha_indexes(img.dataset)
            b_count += len(idx)
        return b_count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if not self.closed:
            self.close()

    def close(self):
        for img in self.imageReaders:
            img.close()
        self.closed = True

    @property
    def profile(self):
        dataset = self.imageReaders[0].dataset
        width, height = dataset.width, dataset.height
        src_transform = dataset.transform

        src_profile = dict(
            driver="GTiff",
            dtype=dataset.dtypes[0],
            height=height,
            width=width,
            nodata=dataset.nodata,
            crs=dataset.crs,
            transform=src_transform,
            compress='RAW',
        )
        return src_profile

    def get_block_windows(self, size=2048):
        dataset = self.imageReaders[0].dataset
        width = dataset.width
        height = dataset.height
        i_len = height // size + 1
        j_len = width // size + 1
        blocks = []
        for i in range(i_len):
            h = size if i != i_len - 1 else height % size
            for j in range(j_len):
                w = size if j != j_len - 1 else width % size
                blocks.append(Window(j * size, i * size, w, h))
        return blocks

    def buffer_block(self, window, buffer):
        col = window.col_off - buffer
        row = window.row_off - buffer
        width = window.width + 2 * buffer
        height = window.height + 2 * buffer
        if col < 0:
            col = 0
        if row < 0:
            row = 0
        reader = self.imageReaders[0].dataset
        if width + col > reader.width:
            width = reader.width - col
        if height + row > reader.height:
            height = reader.height - row
        buffered = Window(col, row, width, height)
        shrink = Window(window.col_off - col, window.row_off - row, window.width, window.height)
        return buffered, shrink

    def multi_execute(self, execs, idx=None, buffer=0, notify=None, size=2048):
        if len(execs) == 0:
            return
        if idx is None:
            idx = []
        empty_idx = len(idx) == 0
        blocks = self.get_block_windows(size)
        rd = self.imageReaders[0].dataset
        total = rd.width * rd.height
        current = 0
        for window in blocks:
            if buffer > 0:
                buffered, shrink = self.buffer_block(window, buffer)
            else:
                buffered = window
                shrink = Window(0, 0, window.width, window.height)
            image_block = []
            for image in self.imageReaders:
                if empty_idx:
                    idx = tuple([n + 1 for n in range(image.dataset.count)])
                data = image.dataset.read(idx, window=buffered).astype('float32')
                image_block.extend(data)
            np_data = np.array(image_block)
            mask = (np_data == 0)
            for kernel in execs:
                kernel.run(np_data, mask, window, shrink)
            current += window.width * window.height
            if notify is not None:
                notify(total, current)

    def part(self, geo_feature):
        cut_line = create_cutline(self.imageReaders[0].dataset, geo_feature, geometry_crs="epsg:4326")
        bbox = feature_bounds(geo_feature)
        data = []
        mask = None
        for img in self.imageReaders:
            da, mask = img.part(bbox, vrt_options={'cutline': cut_line})
            for dx in da:
                data.append(dx)
        return np.array(data), mask

    def tile(self, x, y, z, tilesize=256):
        data = []
        mask = None
        for img in self.imageReaders:
            da, mask = img.tile(x, y, z, tilesize)
            for dx in da:
                data.append(dx)
        return np.array(data), mask

    def read(self, index, window):
        return self.imageReaders[index - 1].dataset.read(1, window=window)


class MorphologyKernel(DaKernel):
    def run(self, data, mask, window, shrink_window=None):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # dilation
        img = cv2.dilate(data, kernel, iterations=1)
        # opening
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)
        # closing
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
        return (img * 2).astype(np.uint8), mask


class DaWriterKernel(DaKernel):
    def __init__(self, destination=[], list_value=None, ids_output=[0], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.destination = destination
        self.profile = kwargs
        self.writer = [None] * len(self.destination)
        for _id, path in enumerate(destination):
            if path is not None:
                writer = rasterio.open(path, mode='w', **kwargs)
                self.writer[_id] = writer

        self.closed = False
        self.list_value = list_value
        self.ids_output = ids_output

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if not self.closed:
            self.close()

    def close(self):
        for writer in self.writer:
            if writer is not None:
                writer.close()
                self.closed = True

    def run(self, data, _, window, shrink_window=None):

        for idx, _da in enumerate(data):
            if shrink_window is not None:
                row_start = shrink_window.row_off
                row_end = shrink_window.row_off + shrink_window.height
                col_start = shrink_window.col_off
                col_end = shrink_window.col_off + shrink_window.width

                _result = _da[row_start:row_end, col_start:col_end]
                if self.list_value is not None:
                    data_tmp = np.zeros(np.shape(_result)).astype(np.uint8)
                    for i, j in enumerate(self.list_value):
                        data_tmp[_result == i + 1] = j
                    data_tmp[_result == 0] = 1
                    _result = data_tmp
                if idx in self.ids_output:
                    self.writer[idx].write(_result.astype(np.uint8), indexes=1, window=window)


class DaSyntheticKernel(DaKernel):
    def __init__(self, kernels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(kernels) == 0:
            raise Exception('Kernels is empty')
        self.kernels = kernels

    def run(self, data, mask, window, shrink_window=None):
        temp, temp_mask = self.kernels[0].run(data, mask, window, shrink_window)
        if len(self.kernels) > 1:
            for i in range(1, len(self.kernels)):
                result = self.kernels[i].run(temp, temp_mask, window, shrink_window)
                if result:
                    temp, temp_mask = result


class DaComputeKernel(DaKernel):
    def __init__(self, callback=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback = callback
        self.init()

    def init(self):
        pass

    def run(self, data, mask, window, shrink_window=None):
        return self.callback(data, mask, window, shrink_window)


class DaStretchKernel(DaKernel):
    def __init__(self, qt_scheme=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qt_scheme = qt_scheme

    def run(self, data, mask, window, shrink_window=None):
        ids = range(len(data))
        window_image = data
        cut_shape = np.shape(window_image)
        new_image = np.zeros((cut_shape[1], cut_shape[2], len(ids)), dtype=np.uint8)
        for i in ids:
            band = window_image[i]
            try:
                band_qt = self.qt_scheme[i]
                cut_nor = np.interp(band, (band_qt.get('p2'), band_qt.get('p98')), (1, 255)).astype(np.uint8)
                # nodatamask = (band == np.uint8(0))
            except Exception:
                cut_nor = band.astype(np.uint8)
            band[mask[0]] = 0
            new_image[..., i] = cut_nor

        result = new_image / 255.0
        return result, mask


class DaUnetPredictKernel(DaKernel):
    def __init__(self, model=None, unet_type=None, size=None, padding=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.unet_type = unet_type
        self.size = size
        self.padding = padding

    def run(self, data, mask, window, shrink_window=None):
        input_image = np.zeros((self.size, self.size, data.shape[2]))
        col = shrink_window.col_off - self.padding
        if col < 0:
            col = 0
        row = shrink_window.row_off - self.padding
        if row < 0:
            row = 0

        input_image[row:data.shape[0], col:data.shape[1], :] = data
        X = np.array([input_image])

        preds = self.model.predict(X, verbose=1)
        num_class = self.model.output[-1].shape[-1]
        if not isinstance(preds, list):
            preds = [preds]

        predictdata = []

        for _pred in preds:
            if num_class == 1:
                predict = (_pred > 0.5).astype(np.uint8)
                predict_data = predict[0, row:data.shape[0], col:data.shape[1], 0]
            else:
                predict_data = np.argmax(_pred[0, row:data.shape[0], col:data.shape[1], :], axis=2)

            try:
                mask_data = mask[0, row:data.shape[0], col:data.shape[1]]
                predict_data[mask_data] = 0
            except:
                mask_data = mask[0, row:data.shape[0], col:data.shape[1]]
                predictdata[np.transpose(mask_data)] = 0
            predictdata.append(predict_data)

        return predictdata, mask
