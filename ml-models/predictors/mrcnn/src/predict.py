from params import TMP_PATH
from utils.image import remove_nan_value
from utils.image import get_quantile_schema
from mrcnn_processor import MrcnnRunning
from config import Config
import os
import cv2
import sys
import numpy as np
from tqdm import *
import cv2
import gdal
import rasterio
import tensorflow as tf
from shapely.geometry import Polygon
import uuid
ROOT_DIR = os.getcwd()

sys.path.append(ROOT_DIR)


def make_temp_folder():
    temp_folder = uuid.uuid4().hex
    cache_dir = TMP_PATH
    path = '{}/{}'.format(cache_dir, temp_folder)
    os.makedirs(path)
    return path


class PredictConfig(Config):
    def __init__(self, **kwargs):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        self.MEAN_PIXEL = kwargs.get('MEAN_PIXEL')
        self.IMAGES_PER_GPU = kwargs.get("IMAGES_PER_GPU") or 1
        self.IMAGE_CHANNEL_COUNT = kwargs.get("IMAGE_CHANNEL_COUNT") or 3
        self.MAX_GT_INSTANCES = kwargs.get("MAX_GT_INSTANCES") or 500
        self.ROI_POSITIVE_RATIO = kwargs.get("ROI_POSITIVE_RATIO") or 0.66

        # self.NUM_CLASSES = kwargs.get("NUM_CLASSES") or (1 + 1)  # 1 Backgroun + 1 Building
        print('model parameter:', kwargs)

        self.trainer_size = int(kwargs.get('trainer_size')) or 512
        self.IMAGE_MAX_DIM = kwargs.get(
            "IMAGE_MAX_DIM") or (self.trainer_size + 64)
        self.IMAGE_MIN_DIM = kwargs.get(
            "IMAGE_MIN_DIM") or (self.trainer_size + 64)
        super().__init__()

        self.DETECTION_MAX_INSTANCES = kwargs.get(
            "DETECTION_MAX_INSTANCES") or 300
        self.DETECTION_NMS_THRESHOLD = kwargs.get(
            "DETECTION_NMS_THRESHOLD") or 0.3
        self.DETECTION_MIN_CONFIDENCE = kwargs.get(
            "DETECTION_MIN_CONFIDENCE") or 0.6

        RPN_ANCHOR_SCALES = None
        try:
            RPN_ANCHOR_SCALES = tuple(kwargs.get(
                'param').get("RPN_ANCHOR_SCALES"))
        except Exception:
            pass
        self.RPN_ANCHOR_SCALES = RPN_ANCHOR_SCALES or (32, 64, 128, 256, 512)
        self.RPN_NMS_THRESHOLD = kwargs.get("RPN_NMS_THRESHOLD") or 0.7


class MrcnnPredictor(MrcnnRunning):
    def __init__(self, mode, image_path, model_path, **kwargs):
        self.model_dir = os.path.dirname(model_path)
        self.config = PredictConfig(**kwargs)

        super().__init__(mode, self.config, self.model_dir)
        self.image_path = image_path
        self.model_path = model_path

        self.numbands_predict = kwargs.get('numbands_predict')
        self.input_size = kwargs.get('predict_size') or 512
        # cut_size = kwargs.get('cut_size') or int(self.crop_size -50)
        with rasterio.open(self.image_path) as src:
            self.image_dtype = src.dtypes[0]
        self.config.display()

    def check_intersec(self, contour, input_size, cut_size):
        padding = int((input_size - cut_size)/2)
        cnt1 = np.array([[padding, padding], [padding, input_size-padding],
                        [input_size-padding, input_size-padding], [input_size-padding, padding]])
        contour1 = np.array(cnt1.reshape(-1, 1, 2), dtype=np.int32)
        img1 = np.zeros((input_size, input_size)).astype(np.uint8)
        img1 = cv2.fillConvexPoly(img1, contour1, 255)
        img = np.zeros((input_size, input_size)).astype(np.uint8)
        img = cv2.fillConvexPoly(img, contour, 255)
        img_result = cv2.bitwise_and(img1, img)
        contours_rs, _ = cv2.findContours(
            img_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        try:
            if cv2.contourArea(contours_rs[0])/float(cv2.contourArea(contour)) > 0.50:
                return True
            else:
                return False
        except Exception:
            return False

    def nms_result(self, list_polygons, list_scores, iou_threshold=0.4):
        list_shapely_polygons = [Polygon(polygon) for polygon in list_polygons]
        list_bound = [np.array(polygon.bounds)
                      for polygon in list_shapely_polygons]
        result = tf.image.non_max_suppression(np.array(list_bound), np.array(
            list_scores), 1000000, iou_threshold=iou_threshold)
        with tf.Session() as sess:
            out = sess.run([result])
            indexes = out[0]
        result_polygons = [list_polygons[idx] for idx in indexes]
        result_scores = [list_scores[idx] for idx in indexes]
        return result_polygons, result_scores

    def predict_image(self, cut_size):
        dataset_image = gdal.Open(self.image_path)
        w, h = dataset_image.RasterXSize, dataset_image.RasterYSize
        num_band = self.numbands_predict or dataset_image.RasterCount
        temp_folder = make_temp_folder()
        new_image = f'{temp_folder}/new_image.tif'
        remove_nan_value(self.image_path, new_image)
        self.image_path = new_image
        qt_schema = self._get_quantile_schema()

        model = self.identify_model()
        self.load_weights(self.model_path, by_name=True)

        if h <= self.input_size or w <= self.input_size:
            raw_image = dataset_image.ReadAsArray()[0:num_band]
            if len(raw_image.shape) == 2:
                raw_image = raw_image[np.newaxis, ...]
            image = np.zeros(
                (num_band, raw_image.shape[1], raw_image.shape[2]))
            for idx in range(num_band):
                band = raw_image[idx]
                band_qt = qt_schema[idx] if qt_schema else None
                if band_qt:
                    stretched_band = np.interp(
                        band, (band_qt['p2'], band_qt['p98']), (1, 255)).astype(np.uint8)
                else:
                    stretched_band = band
                image[idx] = stretched_band
            image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
            image_res = 0.3

            self.config.IMAGE_MAX_DIM = (
                round(max(h, w) / 64 * 3.2 / 3 * image_res / 0.3)) * 64
            self.config.IMAGE_MIN_DIM = (
                round(min(h, w) / 64 * 3.2 / 3 * image_res / 0.3)) * 64
            self.config.display()

            predictions = self.detect(
                [image] * self.config.BATCH_SIZE, verbose=1)
            p = predictions[0]
            list_contours = []
            list_score = []

            for i in range(p['masks'].shape[2]):
                try:
                    mask = p['masks'][:, :, i].astype(np.uint8)
                    score = p['scores'][i]
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cv2.contourArea(contours[0]) > 100:
                        list_contours.append(contours[0])
                        list_score.append(score)
                except Exception:
                    pass
            return_contour = list_contours
            predictions = None
            p = None
            model = None
        else:

            self.config.display()
            return_contour = []
            list_score = []

            padding = int((self.input_size - cut_size) / 2)
            new_w = w + 2 * padding
            new_h = h + 2 * padding
            cut_w = list(range(padding, new_w - padding, cut_size))
            cut_h = list(range(padding, new_h - padding, cut_size))
            list_hight = []
            list_weight = []
            for i in cut_h:
                list_hight.append(i)

            for i in cut_w:
                list_weight.append(i)

            with tqdm(total=len(list_hight) * len(list_weight)) as pbar:
                for i in range(len(list_hight)):
                    # on_processing(float(i) * 0.9 / len(list_hight))
                    start_y = list_hight[i]
                    for j in range(len(list_weight)):
                        start_x = list_weight[j]
                        startx = start_x - padding
                        endx = min(start_x + cut_size +
                                   padding, new_w - padding)

                        starty = start_y - padding
                        endy = min(start_y + cut_size +
                                   padding, new_h - padding)

                        if startx == 0:
                            xoff = startx
                        elif startx > endx - padding:
                            xoff = endx - padding
                        else:
                            xoff = startx - padding

                        if starty == 0:
                            yoff = starty
                        elif starty > endy - padding:
                            yoff = endy - padding
                        else:
                            yoff = starty - padding

                        xcount = endx - padding - xoff
                        ycount = endy - padding - yoff

                        try:
                            raw_image = dataset_image.ReadAsArray(
                                xoff, yoff, xcount, ycount)
                            if len(raw_image.shape) == 2:
                                raw_image = raw_image[np.newaxis, ...]
                            image = np.zeros(
                                (num_band, raw_image.shape[1], raw_image.shape[2]))
                            for idx in range(num_band):
                                band = raw_image[idx]
                                band_qt = qt_schema[idx] if qt_schema else None
                                if band_qt:
                                    stretched_band = np.interp(band, (band_qt['p2'], band_qt['p98']), (1, 255)).astype(
                                        np.uint8)
                                else:
                                    stretched_band = band
                                image[idx] = stretched_band
                            image_detect = np.transpose(
                                image, (1, 2, 0)).astype(np.uint8)
                        except Exception as e:
                            print(e)
                            image_detect = np.zeros(
                                (self.input_size, self.input_size, num_band)).astype(np.uint8)

                        if image_detect.shape[0] < self.input_size or image_detect.shape[1] < self.input_size:
                            img_temp = np.zeros(
                                (self.input_size, self.input_size, num_band))
                            if startx == 0 and starty == 0:
                                img_temp[(self.input_size - image_detect.shape[0]):self.input_size,
                                         (self.input_size - image_detect.shape[1]):self.input_size] = image_detect
                            elif startx == 0:
                                img_temp[0:image_detect.shape[0],
                                         (self.input_size - image_detect.shape[1]):self.input_size] = image_detect
                            elif starty == 0:
                                img_temp[(self.input_size - image_detect.shape[0]):self.input_size,
                                         0:image_detect.shape[1]] = image_detect
                            else:
                                img_temp[0:image_detect.shape[0],
                                         0:image_detect.shape[1]] = image_detect

                            image_detect = img_temp
                        if np.count_nonzero(image_detect) > 0:
                            image_detect = image_detect.astype(np.uint8)
                            predictions = self.detect(
                                [image_detect] * self.config.BATCH_SIZE, verbose=1)

                            p = predictions[0]

                            list_temp = []
                            list_temp_score = []

                        # ##########################################################
                            for i in range(p['masks'].shape[2]):
                                mask = p['masks'][:, :, i].astype(np.uint8)
                                score = p["scores"][i]
                                contours, _ = cv2.findContours(
                                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                try:
                                    if cv2.contourArea(contours[0]) > 10:
                                        if (contours[0].max() < (self.input_size-padding)) and (contours[0].min() > padding):
                                            # print(1)
                                            list_temp.append(contours[0])
                                            list_temp_score.append(score)
                                        elif (contours[0].max() < (self.input_size-5)) and (contours[0].min() > 5):
                                            list_temp.append(contours[0])
                                            list_temp_score.append(score)
                                except Exception:
                                    pass

                        #########################################################
                            temp_contour = []
                            for contour in list_temp:
                                anh = contour.reshape(-1, 2)
                                anh2 = anh + \
                                    np.array(
                                        [startx - padding, starty - padding])
                                con_rs = anh2.reshape(-1, 1, 2)
                                temp_contour.append(con_rs)

                            return_contour.extend(temp_contour)
                            list_score.extend(list_temp_score)
                        pbar.update()
                predictions = None
                p = None
                model = None
                list_contours = return_contour
        return return_contour, list_score

        # cuda.select_device(0)
        # cuda.close()
    def predict(self):
        # list_size_crop = [int(self.input_size*3/4), int(self.input_size*7/8), int(self.input_size*5/8)]
        list_size_crop = [int(self.input_size*3/4), int(self.input_size*7/8)]
        result_contours = []
        result_scores = []
        for size_crop in list_size_crop:
            # out_path_tmp = os.path.join(dir_tmp, str(size_crop) + '.geojson')
            list_contours, list_score = self.predict_image(size_crop)
            result_contours.extend(list_contours)
            result_scores.extend(list_score)

        result_polygons = [
            result_contours[i].reshape(-1, 2) for i in range(len(result_contours))]
        polygon_result_nms, score_result_nms = self.nms_result(
            result_polygons, result_scores)
        return polygon_result_nms, score_result_nms

    def _get_quantile_schema(self):
        if self.image_dtype == 'uint8':
            return None
        return get_quantile_schema(self.image_path)
