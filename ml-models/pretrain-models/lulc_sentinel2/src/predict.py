import numpy as np
import rasterio
from rasterio.windows import Window
import threading
from tqdm import tqdm
import concurrent.futures
import warnings, cv2
import tensorflow as tf
from .utils import get_nor_val
from tensorflow.keras.models import load_model
import os

warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
tf.compat.v1.Session(config=config)

def predict(path_image, path_predict, size=256, list_models=None, numbands=6):
    bareland_model=list_models[0]
    crops_model=list_models[1]
    forest_model=list_models[2]
    builtup_model=list_models[3]
    water_model=list_models[4]
    green_model=list_models[5]
    print(path_image)

    with rasterio.open(path_image) as raster:
        meta = raster.meta
        meta.update({'count': 1, 'nodata': 0,"dtype":"uint8"})
        height, width = raster.height, raster.width
        input_size = size
        stride_size = input_size - input_size //4
        padding = int((input_size - stride_size) / 2)
        
        list_coordinates = []
        for start_y in range(0, height, stride_size):
            for start_x in range(0, width, stride_size): 
                x_off = start_x if start_x==0 else start_x - padding
                y_off = start_y if start_y==0 else start_y - padding
                    
                end_x = min(start_x + stride_size + padding, width)
                end_y = min(start_y + stride_size + padding, height)
                
                x_count = end_x - x_off
                y_count = end_y - y_off
                list_coordinates.append(tuple([x_off, y_off, x_count, y_count, start_x, start_y]))

        path_tmp_pre = path_predict.replace(".tif","_tmp.tif")
        with rasterio.open(path_tmp_pre,'w+', **meta, compress='lzw') as r:
            read_lock = threading.Lock()
            write_lock = threading.Lock()

            def predict_class(img_pre,class_model,mask, shape, thres=0.5):
                if type(class_model.output) is list:
                    y_pred = class_model.predict(img_pre)[-1]
                else:
                    y_pred = class_model.predict(img_pre)
                y_pred = (y_pred[0,...,0] > thres).astype(np.uint8)
                y = y_pred[mask].reshape(shape)
                y = morphology(y)
                return y

            def process(coordinates):
                x_off, y_off, x_count, y_count, start_x, start_y = coordinates
                read_wd = Window(x_off, y_off, x_count, y_count)
                with read_lock:
                    if raster.count == 10:
                        values = raster.read((1,2,3,8,9,10),window=read_wd)
                    if raster.count == 12:
                        values = raster.read((1,2,3,7,8,9),window=read_wd)
                if raster.profile["dtype"]=="uint8":
                    image_detect = values[0:3].transpose(1,2,0).astype(int)
                else:
                    datas = get_nor_val(values)
                    image_detect = np.transpose(datas, (1,2,0))

                img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                mask = np.pad(np.ones((stride_size, stride_size), dtype=np.uint8), ((padding, padding),(padding, padding)))
                shape = (stride_size, stride_size)
                if y_count < input_size or x_count < input_size:
                    img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                    mask = np.zeros((input_size, input_size), dtype=np.uint8)
                    if start_x == 0 and start_y == 0:
                        img_temp[(input_size - y_count):input_size, (input_size - x_count):input_size] = image_detect
                        mask[(input_size - y_count):input_size-padding, (input_size - x_count):input_size-padding] = 1
                        shape = (y_count-padding, x_count-padding)
                    elif start_x == 0:
                        img_temp[0:y_count, (input_size - x_count):input_size] = image_detect
                        if y_count == input_size:
                            mask[padding:y_count-padding, (input_size - x_count):input_size-padding] = 1
                            shape = (y_count-2*padding, x_count-padding)
                        else:
                            mask[padding:y_count, (input_size - x_count):input_size-padding] = 1
                            shape = (y_count-padding, x_count-padding)
                    elif start_y == 0:
                        img_temp[(input_size - y_count):input_size, 0:x_count] = image_detect
                        if x_count == input_size:
                            mask[(input_size - y_count):input_size-padding, padding:x_count-padding] = 1
                            shape = (y_count-padding, x_count-2*padding)
                        else:
                            mask[(input_size - y_count):input_size-padding, padding:x_count] = 1
                            shape = (y_count-padding, x_count-padding)
                    else:
                        img_temp[0:y_count, 0:x_count] = image_detect
                        mask[padding:y_count, padding:x_count] = 1
                        shape = (y_count-padding, x_count-padding)
                        
                    image_detect = img_temp
                mask = (mask!=0)

                if np.count_nonzero(image_detect) > 0:
                    if len(np.unique(image_detect)) <= 2:
                        pass
                    else:

                        img_pre = image_detect[np.newaxis,...]

                        y_water = predict_class(img_pre, water_model, mask, shape, thres=0.8)
                        y_builtup = predict_class(img_pre, builtup_model, mask, shape, thres=0.7)
                        y_crops = predict_class(img_pre, crops_model, mask, shape)
                        y_forest = predict_class(img_pre, forest_model, mask, shape)
                        y_bare = predict_class(img_pre, bareland_model, mask, shape)

                        y_green = predict_class(img_pre, green_model, mask, shape)

                        
                        y_crops = y_crops*y_green
                        y_forest = y_forest*y_green

                        y_bare = y_bare*(1-y_green)

                        y_final = np.zeros(shape)
                        y_final[y_bare>0]=1
                        y_final[y_green>0]=2
                        y_final[y_forest>0]=3
                        y_final[y_crops>0]=4
                        y_final[y_builtup>0]=5
                        y_final[y_water>0]=6

                        with write_lock:
                            r.write(y_final[np.newaxis,...], window=Window(start_x, start_y, shape[1], shape[0]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                results = list(tqdm(executor.map(process, list_coordinates), total=len(list_coordinates)))


        list_coords = []
        for start_y in range(0, height, stride_size):
            for start_x in range(0, width, stride_size): 
                x_count = min(stride_size, width-start_x)
                y_count = min(stride_size, height-start_y)
                list_coords.append(tuple([start_x, start_y, x_count, y_count]))

        with rasterio.open(path_tmp_pre) as tmp_predict:
            with rasterio.open(path_predict,'w+', **meta, compress='lzw') as r:

                read_lock = threading.Lock()
                write_lock = threading.Lock()
                def remove_cloud(coordinates):
                    start_x, start_y, x_count, y_count = coordinates
                    read_wd = Window(start_x, start_y, x_count, y_count)
 
                    with read_lock:
                        values = raster.read(window=read_wd)
                        mask_val = tmp_predict.read(window=read_wd)
                    
                    mask_nocloud = np.mean(values, axis=0) < 4000
                    mask_nocloud_int = mask_nocloud.astype('uint8')
                    new_mask_val = mask_val * mask_nocloud_int
                    with write_lock:
                        r.write(new_mask_val, window=Window(start_x, start_y, x_count, y_count))
                        r.write_colormap(
                                1, {
                                    1: (165,155,143),
                                    2: (9,250,66),
                                    3:(57,125,73),
                                    4:(228,150,53),
                                    5:(196,40,27),
                                    6:(65,155,223),
                                    })
                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                    results1 = list(tqdm(executor.map(remove_cloud, list_coords), total=len(list_coords)))
        try:
            os.remove(tmp_predict)
        except:
            pass
                    
def morphology(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)
    return img
    
if __name__=="__main__":
    image_path = '/home/skymap/BoVu/LULC/images_6band/test.tif'
    output_tif = '/home/skymap/BoVu/LULC/tmp/test.tif'

    green_model = load_model('/home/skymap/BoVu/LULC/models_LULC_indo/green.h5', compile=False)
    bareland_model = load_model('/home/skymap/BoVu/LULC/models_LULC_indo/bareland.h5', compile=False)
    crops_model = load_model('/home/skymap/BoVu/LULC/models_LULC_indo/crops.h5', compile=False)
    forest_model = load_model('/home/skymap/BoVu/LULC/models_LULC_indo/forest.h5', compile=False)
    builtup_model = load_model('/home/skymap/BoVu/LULC/models_LULC_indo/builtup.h5', compile=False)
    water_model = load_model('/home/skymap/BoVu/LULC/models_LULC_indo/water.h5', compile=False)

    list_models = [bareland_model,crops_model,forest_model,builtup_model,water_model,green_model]
    predict(image_path, output_tif, size=128, list_models=list_models, numbands=6)
