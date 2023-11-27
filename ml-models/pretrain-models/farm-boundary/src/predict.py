
import rasterio
from vectorization import Vectorization
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects
from rasterio.windows import Window
from tqdm import tqdm
import numpy as np
import cv2


def raster_to_vector(path_in, path_out, threshold_distance=3, threshold_connect=5):
    print('start convert raster to vector ...')
    with rasterio.open(path_in) as inds:
        data = inds.read()[0]
        transform = inds.transform
        projstr = inds.crs.to_string()

    data = morphology(data)
    data = remove_small_holes(data.astype(bool), area_threshold=77)
    data = remove_small_objects(data, min_size=77)
    skeleton = skeletonize(data.astype(np.uint8))

    Vectorization.save_polygon(np.pad(skeleton, pad_width=1).astype(np.intc), threshold_distance, threshold_connect,
                               transform, projstr, path_out)
    print("Done!!!")


def morphology(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # dilation
    # img = cv2.dilate(data,kernel,iterations = 1)
    # opening
    #     img = cv2.morphologyEx(data, cv2.MORPH_OPEN, kernel)
    # for i in range(10):
    #     img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)
    # closing
    #     for _ in range(2):
    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
    return img


def predict_farm(model, path_image, path_predict, size=480):
    with rasterio.open(path_image) as raster:
        meta = raster.meta
        meta.update({'count': 1, 'nodata': 0})
        height, width = raster.height, raster.width
        input_size = size
        stride_size = input_size - input_size // 4
        padding = int((input_size - stride_size) / 2)

        list_coordinates = []
        for start_y in range(0, height, stride_size):
            for start_x in range(0, width, stride_size):
                x_off = start_x if start_x == 0 else start_x - padding
                y_off = start_y if start_y == 0 else start_y - padding

                end_x = min(start_x + stride_size + padding, width)
                end_y = min(start_y + stride_size + padding, height)

                x_count = end_x - x_off
                y_count = end_y - y_off
                list_coordinates.append(tuple([x_off, y_off, x_count, y_count, start_x, start_y]))

        with tqdm(total=len(list_coordinates)) as pbar:
            with rasterio.open(path_predict, 'w+', **meta, compress='lzw') as r:
                for x_off, y_off, x_count, y_count, start_x, start_y in list_coordinates:
                    image_detect = raster.read(window=Window(x_off, y_off, x_count, y_count))[0:3].transpose(1, 2, 0)
                    mask = np.pad(np.ones((stride_size, stride_size), dtype=np.uint8),
                                  ((padding, padding), (padding, padding)))
                    shape = (stride_size, stride_size)
                    if y_count < input_size or x_count < input_size:
                        img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                        mask = np.zeros((input_size, input_size), dtype=np.uint8)
                        if start_x == 0 and start_y == 0:
                            img_temp[(input_size - y_count):input_size,
                            (input_size - x_count):input_size] = image_detect
                            mask[(input_size - y_count):input_size - padding,
                            (input_size - x_count):input_size - padding] = 1
                            shape = (y_count - padding, x_count - padding)
                        elif start_x == 0:
                            img_temp[0:y_count, (input_size - x_count):input_size] = image_detect
                            if y_count == input_size:
                                mask[padding:y_count - padding, (input_size - x_count):input_size - padding] = 1
                                shape = (y_count - 2 * padding, x_count - padding)
                            else:
                                mask[padding:y_count, (input_size - x_count):input_size - padding] = 1
                                shape = (y_count - padding, x_count - padding)
                        elif start_y == 0:
                            img_temp[(input_size - y_count):input_size, 0:x_count] = image_detect
                            if x_count == input_size:
                                mask[(input_size - y_count):input_size - padding, padding:x_count - padding] = 1
                                shape = (y_count - padding, x_count - 2 * padding)
                            else:
                                mask[(input_size - y_count):input_size - padding, padding:x_count] = 1
                                shape = (y_count - padding, x_count - padding)
                        else:
                            img_temp[0:y_count, 0:x_count] = image_detect
                            mask[padding:y_count, padding:x_count] = 1
                            shape = (y_count - padding, x_count - padding)

                        # image_detect = np.array((img_temp/127.5) - 1, dtype=np.float32)
                        image_detect = img_temp
                    mask = (mask != 0)

                    if np.count_nonzero(image_detect) > 0:
                        if len(np.unique(image_detect)) <= 2:
                            pass
                        else:
                            y_pred = model.predict(image_detect[np.newaxis, ...] / 255.)[0]
                            y_pred = (y_pred[0, ..., 0] > 0.5).astype(np.uint8)
                            y = y_pred[mask].reshape(shape)
                            r.write(y[np.newaxis, ...], window=Window(start_x, start_y, shape[1], shape[0]))
                    pbar.update()
