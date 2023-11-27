from config import Config
import numpy as np
import cv2
from osgeo import gdal
import model as modellib
from tqdm import tqdm
import geopandas as gpd
import tensorflow as tf
from glob import glob
from utils.export_data import exportResult2
from utils.convert_datatype import list_contour_to_list_polygon
from osgeo import osr, gdal
import os


class TreeCounting2StrideV2():
    MODEL_DIR = './'

    class InferenceConfig(Config):
        image_size = 256
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 1  # 1 Background + 1 Building
        IMAGE_MAX_DIM = image_size + 64
        IMAGE_MIN_DIM = image_size + 64
        DETECTION_MAX_INSTANCES = 200
        MAX_GT_INSTANCES = 200

        MASK_SHAPE = [28, 28]
        USE_MINI_MASK = True
        MINI_MASK_SHAPE = (56, 56)
        NAME = "crowdai-mapping-challenge"

        DETECTION_NMS_THRESHOLD = 0.4
        DETECTION_MIN_CONFIDENCE = 0.6
        MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
        IMAGE_CHANNEL_COUNT = 3
        RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    def check_intersec(self, contour, input_size, overlapsize):
        padding = int((input_size - overlapsize) / 2)
        cnt1 = np.array(
            [[padding, padding], [padding, input_size - padding], [input_size - padding, input_size - padding],
             [input_size - padding, padding]])
        contour1 = np.array(cnt1.reshape(-1, 1, 2), dtype=np.int32)
        img1 = np.zeros((input_size, input_size)).astype(np.uint8)
        img1 = cv2.fillConvexPoly(img1, contour1, 255)
        img = np.zeros((input_size, input_size)).astype(np.uint8)
        img = cv2.fillConvexPoly(img, contour, 255)
        img_result = cv2.bitwise_and(img1, img)
        contours_rs, hierarchy = cv2.findContours(
            img_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        try:
            if cv2.contourArea(contours_rs[0]) / float(cv2.contourArea(contour)) > 0.50:
                return True
            else:
                return False
        except Exception:
            return False

    def predict_large_img(self, image_path, model_path, crop_size, output_path, option=None):
        dataset_image = gdal.Open(image_path)
        w, h = dataset_image.RasterXSize, dataset_image.RasterYSize
        num_band = 3
        input_size = 256
        if h <= input_size or w <= input_size:
            image = dataset_image.ReadAsArray()[0:num_band].swapaxes(
                0, 1).swapaxes(1, 2).astype(np.uint8)
            image_res = 0.3
            config = self.InferenceConfig()
            config.IMAGE_MAX_DIM = (
                round(max(h, w) / 64 * 3.2 / 3 * image_res / 0.3)) * 64
            config.IMAGE_MIN_DIM = (
                round(min(h, w) / 64 * 3.2 / 3 * image_res / 0.3)) * 64
            # config = InferenceConfig()
            config.display()

            model = modellib.MaskRCNN(
                mode="inference", model_dir=self.MODEL_DIR, config=config)

            print("inside predict api")
            print(model_path)
            model.load_weights(model_path, by_name=True, exclude='conv1')

            class_names = ['BG', 'building']
            predictions = model.detect([image] * config.BATCH_SIZE, verbose=1)
            p = predictions[0]
            boxes = p['rois']
            N = boxes.shape[0]
            list_contours = []
            for i in range(N):
                if not np.any(boxes[i]):
                    continue
                y1, x1, y2, x2 = boxes[i]
                contour = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
                contour = contour.reshape(-1, 1, 2)
                try:
                    if cv2.contourArea(contour) > 100:
                        list_contours.append(contour)
                except Exception:
                    pass
            return_contour = list_contours
            predictions = None
            p = None
            model = None
        else:
            config = self.InferenceConfig()
            config.display()

            model = modellib.MaskRCNN(
                mode="inference", model_dir=self.MODEL_DIR, config=config)
            model.load_weights(model_path, by_name=True)
            class_names = ['BG', 'change']
            return_contour = []
            padding = int((input_size - crop_size) / 2)
            new_w = w + 2 * padding
            new_h = h + 2 * padding
            cut_w = list(range(padding, new_w - padding, crop_size))
            cut_h = list(range(padding, new_h - padding, crop_size))
            list_hight = []
            list_weight = []
            print(w, h)
            for i in cut_h:
                list_hight.append(i)

            for i in cut_w:
                list_weight.append(i)

            with tqdm(total=len(list_hight) * len(list_weight)) as pbar:
                for i in range(len(list_hight)):
                    start_y = list_hight[i]
                    for j in range(len(list_weight)):
                        start_x = list_weight[j]
                        startx = start_x - padding
                        endx = min(start_x + crop_size +
                                   padding, new_w - padding)
                        starty = start_y - padding
                        endy = min(start_y + crop_size +
                                   padding, new_h - padding)
                        if startx == 0:
                            xoff = startx
                        else:
                            xoff = startx - padding
                        if starty == 0:
                            yoff = starty
                        else:
                            yoff = starty - padding
                        xcount = endx - padding - xoff
                        ycount = endy - padding - yoff

                        if num_band == 1:
                            inage_detect = dataset_image.ReadAsArray(
                                xoff, yoff, xcount, ycount)
                            inage_detect = inage_detect[..., np.newaxis]
                        else:
                            inage_detect = dataset_image.ReadAsArray(xoff, yoff, xcount, ycount)[0:num_band].swapaxes(0,
                                                                                                                      1).swapaxes(
                                1, 2)
                        if inage_detect.shape[0] < input_size or inage_detect.shape[1] < input_size:
                            img_temp = np.zeros(
                                (input_size, input_size, inage_detect.shape[2]))
                            if (startx == 0 and starty == 0):
                                img_temp[(input_size - inage_detect.shape[0]):input_size,
                                         (input_size - inage_detect.shape[1]):input_size] = inage_detect
                            elif startx == 0:
                                img_temp[0:inage_detect.shape[0],
                                         (input_size - inage_detect.shape[1]):input_size] = inage_detect
                            elif starty == 0:
                                img_temp[(input_size - inage_detect.shape[0]):input_size,
                                         0:inage_detect.shape[1]] = inage_detect
                            else:
                                img_temp[0:inage_detect.shape[0],
                                         0:inage_detect.shape[1]] = inage_detect
                            inage_detect = img_temp
                        if np.count_nonzero(inage_detect) > 0:
                            predictions = model.detect(
                                [inage_detect] * config.BATCH_SIZE, verbose=0)
                            p = predictions[0]
                            boxes = p['rois']
                            N = boxes.shape[0]
                            list_temp = []
                            for i in range(N):
                                if not np.any(boxes[i]):
                                    continue
                                y1, x1, y2, x2 = boxes[i]
                                contour = np.array(
                                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
                                contour = contour.reshape(-1, 1, 2)
                                try:
                                    if cv2.contourArea(contour) > 10:
                                        if (contour.max() < (input_size - padding)) and (contour.min() > padding):
                                            # print(1)
                                            list_temp.append(contour)
                                        elif (contour.max() < (input_size - 5)) and (
                                                contour.min() > 5) and self.check_intersec(contour, input_size,
                                                                                           crop_size):
                                            list_temp.append(contour)
                                except Exception:
                                    pass
                            temp_contour = []
                            for contour in list_temp:
                                anh = contour.reshape(-1, 2)
                                anh2 = anh + \
                                    np.array(
                                        [startx - padding, starty - padding])
                                con_rs = anh2.reshape(-1, 1, 2)
                                temp_contour.append(con_rs)
                            # print(temp_contour)
                            return_contour.extend(temp_contour)
                        pbar.update()
                predictions = None
                p = None
                model = None
                list_contours = return_contour
        # driverName = "GeoJson"
        driverName = "ESRI Shapefile"
        outputFileName = output_path
        geotransform = dataset_image.GetGeoTransform()
        projection = osr.SpatialReference(dataset_image.GetProjectionRef())
        print(projection)
        polygons_result = list_contour_to_list_polygon(list_contours)
        exportResult2(polygons_result, geotransform,
                      projection, outputFileName, driverName)
        return len(polygons_result)

    def get_list_name_file(self, path_dir, name_file='*.tif'):
        list_file_dir = []
        for file_ in glob(os.path.join(path_dir, name_file)):
            # head, tail = os.path.split(file_)
            list_file_dir.append(file_)
        return list_file_dir

    def nms_shp(self, dir_shp, out_file):
        list_path_file = self.get_list_name_file(dir_shp, '*.shp')
        df1 = gpd.read_file(list_path_file[0])
        df2 = gpd.read_file(list_path_file[1])
        df = df1.append(df2)
        tmp_file = os.path.join(dir_shp, '_tmp.shp')
        df.to_file(tmp_file)
        df = gpd.read_file(tmp_file)
        df_bound = df.bounds
        df_bound = df_bound.to_numpy()
        score = np.random.rand(df_bound.shape[0])
        result = tf.image.non_max_suppression(
            df_bound.tolist(), score.tolist(), 1000000, iou_threshold=0.4)
        with tf.Session() as sess:
            out = sess.run([result])
            indexes = out[0]
        rs_df = df.iloc[indexes]
        rs_df.to_file(out_file)

    def main(self, img_path, model_path, dir_tmp, out_path):
        list_size_crop = [200, 206]
        for size_crop in list_size_crop:
            out_path_tmp = os.path.join(dir_tmp, str(size_crop) + '.shp')
            print(img_path)
            self.predict_large_img(img_path, model_path,
                                   size_crop, out_path_tmp)
        self.nms_shp(dir_tmp, out_path)
