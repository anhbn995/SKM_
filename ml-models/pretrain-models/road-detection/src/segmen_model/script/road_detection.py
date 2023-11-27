from segmen_model.unet_models import models
from rasterio.windows import Window
from tqdm import tqdm
import os
import cv2
import copy
import argparse
import rasterio
import numpy as np
import rasterio.mask
import tensorflow as tf
import skimage.morphology

import warnings
warnings.filterwarnings("ignore")


def main(image_path, result_path, img_size, numband, model_path, num_class, confidence):
    print("*Init model")
    model = models.unet_3plus((img_size, img_size, numband), n_labels=num_class,
                              filter_num_down=[32, 64, 128, 256, 512],
                              filter_num_skip='auto', filter_num_aggregate='auto', stack_num_down=2,
                              stack_num_up=1, activation='ReLU', output_activation='Sigmoid', batch_norm=True,
                              pool=True, unpool=True, deep_supervision=True, multi_input=True, backbone='ResNet50',
                              weights=None, freeze_backbone=False, freeze_batch_norm=True, name='unet3plus')

    model.load_weights(model_path)
    if os.path.exists(result_path):
        pass
    else:
        print(image_path)
        num_band = numband
        input_size = img_size
        current_x, current_y = 0, 0
        stride_size = input_size - 24
        with rasterio.open(image_path) as dataset_image:
            out_meta = dataset_image.meta
            h, w = dataset_image.height, dataset_image.width
            img_1 = np.zeros((h, w))
            list_coordinates = []
            padding = int((input_size - stride_size) / 2)
            new_w = w + 2 * padding
            new_h = h + 2 * padding
            list_weight = list(range(padding, new_w - padding, stride_size))
            list_height = list(range(padding, new_h - padding, stride_size))
            with tqdm(total=len(list_height * len(list_weight))) as pbar:
                for i in range(len(list_height)):
                    top_left_y = list_height[i]
                    for j in range(len(list_weight)):
                        top_left_x = list_weight[j]
                        start_x = top_left_x - padding
                        end_x = min(top_left_x + stride_size +
                                    padding, new_w - padding)
                        start_y = top_left_y - padding
                        end_y = min(top_left_y + stride_size +
                                    padding, new_h - padding)
                        if start_x == 0:
                            x_off = start_x
                        else:
                            x_off = start_x - padding
                        if start_y == 0:
                            y_off = start_y
                        else:
                            y_off = start_y - padding
                        x_count = end_x - padding - x_off
                        y_count = end_y - padding - y_off
                        list_coordinates.append(
                            tuple([x_off, y_off, x_count, y_count, start_x, start_y]))
                        image_detect = dataset_image.read(window=Window(x_off, y_off, x_count, y_count))[
                            :num_band].swapaxes(0, 1).swapaxes(1, 2) / 255.

                        if image_detect.shape[0] < input_size or image_detect.shape[1] < input_size:
                            img_temp = np.zeros(
                                (input_size, input_size, image_detect.shape[2]))
                            if start_x == 0 and start_y == 0:
                                img_temp[(input_size - image_detect.shape[0]):input_size,
                                         (input_size - image_detect.shape[1]):input_size] = image_detect
                            elif start_x == 0:
                                img_temp[0:image_detect.shape[0],
                                         (input_size - image_detect.shape[1]):input_size] = image_detect
                            elif start_y == 0:
                                img_temp[(input_size - image_detect.shape[0]):input_size,
                                         0:image_detect.shape[1]] = image_detect
                            else:
                                img_temp[0:image_detect.shape[0],
                                         0:image_detect.shape[1]] = image_detect
                            image_detect = img_temp
                        if np.count_nonzero(image_detect) > 0:
                            if len(np.unique(image_detect)) == 2 or len(np.unique(image_detect)) == 1:
                                y_pred = image_detect[:, :, 0]
                                pass
                            else:
                                y_pred = model.predict(
                                    tf.expand_dims(image_detect, axis=0))
                                y_pred = y_pred[-1]
                                y_pred = y_pred[0][:, :, 0]
                        else:
                            y_pred = image_detect[:, :, 0]
                            pass
                        if start_x == 0 and start_y == 0:
                            y_pred = y_pred[padding:-padding, padding:-padding]
                        elif start_y == 0 and (x_count + x_off) < w:
                            y_pred = y_pred[padding:-padding, padding:-padding]
                        elif start_y == 0 and (x_count + x_off) >= w:
                            y_pred = y_pred[padding:-padding, padding:x_count]
                        elif (x_count + x_off) >= w and (y_count + y_off) < h:
                            y_pred = y_pred[padding:-padding, padding:x_count]
                        elif start_x == 0 and (y_count + y_off) < h:
                            y_pred = y_pred[padding:-
                                            padding:, padding:-padding]

                        elif start_x == 0 and (y_count + y_off) >= h:
                            y_pred = y_pred[padding:y_count:, padding:-padding]

                        elif (x_count + x_off) >= w and (y_count + y_off) >= h:
                            y_pred = y_pred[padding:y_count, padding:x_count]

                        elif (y_count + y_off) >= h and (x_count + x_off) < w:
                            y_pred = y_pred[padding:y_count, padding:-padding]

                        else:
                            y_pred = y_pred[padding:x_count -
                                            padding:, padding:y_count - padding]

                        if y_pred.shape[1] == 10:
                            pass
                        else:
                            if current_y >= w:
                                current_y = 0
                                current_x = current_x + past_i.shape[0]

                            img_1[current_x:current_x + y_pred.shape[0],
                                  current_y:current_y + y_pred.shape[1]] += y_pred
                            current_y += y_pred.shape[1]
                            past_i = y_pred
                        pbar.update()
    print("*Write image")
    img2 = copy.deepcopy(img_1)
    img2[img2 > confidence] = 0
    img2[img2 != 0] = 1
    img3 = img2.astype(np.bool)
    img3 = skimage.morphology.remove_small_objects(img3, min_size=4128)
    img3 = img3.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    img4 = cv2.dilate(img3, kernel, iterations=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    img4 = cv2.erode(img4, kernel, iterations=3)

    with rasterio.Env():
        profile = out_meta
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw')
    with rasterio.open(result_path, 'w', **profile) as dst:
        dst.write(img4.astype(np.uint8), 1)
    print("*Finished")
    return True


if __name__ == '__main__':
    debug = True
    if not debug:
        args_parser = argparse.ArgumentParser()
        args_parser.add_argument('--image_path', help='Path of image', required=True, typr=str,
                                 default='/home/quyet/WorkSpace/Model/swin_rotation/data_result')
        args_parser.add_argument('--result_path', help='Path of image', required=True, typr=str,
                                 default='/home/quyet/WorkSpace/Model/swin_rotation/data_result')
        args_parser.add_argument(
            '--num_class', help='Number of class', required=False, type=int, default=1)
        args_parser.add_argument(
            '--img_size', help='Image_size', required=False, type=int, default=512)
        args_parser.add_argument(
            '--numband', help='Number band', required=False, type=int, default=3)
        args_parser.add_argument(
            '--confidence', help='Confidence', default=0.999, required=False, type=float)
        args_parser.add_argument('--model_path', help='Path of weights', required=True, default=None,
                                 type=str)
        print("done")
        param = args_parser.parse_args()
        image_path = param.image_path
        result_path = param.result_path
        num_class = param.num_classs
        img_size = param.img_size
        numband = param.numband
        confidence = param.confidence
        model_path = param.model_path
    else:
        image_path = '/home/quyet/data/road/img_resize/Nunukan_1 copy.tif'
        result_path = '/home/quyet/data/road/img_resize'
        num_class = 1
        img_size = 512
        numband = 3
        confidence = 0.999
        model_path = '/home/nghipham/Desktop/project/eof-rq-worker/processors/models/segmen_model/unet_models/weights/unet3plusroad_512_1class_binary.h5'

    main(image_path, result_path, img_size,
         numband, model_path, num_class, confidence)
