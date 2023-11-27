import numpy as np
import rasterio
from rasterio.windows import Window
import threading
from tqdm import tqdm
import concurrent.futures
import warnings
import tensorflow as tf
from tensorflow.keras.models import load_model
from .utils import get_normalized_data

warnings.filterwarnings("ignore")
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# tf.compat.v1.Session(config=config)

def predict(model, cloudy_img, sar_img, path_predict, size=256):
    with rasterio.open(cloudy_img) as raster:
        meta = raster.meta
        with rasterio.open(sar_img) as sar_raster:

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

            with rasterio.open(path_predict,'w+', **meta, compress='lzw') as r:
                read_lock = threading.Lock()
                write_lock = threading.Lock()
                
                def process(coordinates):
                    x_off, y_off, x_count, y_count, start_x, start_y = coordinates
                    read_wd = Window(x_off, y_off, x_count, y_count)
                    with read_lock:
                        cloudy_val = raster.read(window=read_wd).astype('float32')
                        cloudy_val[np.isnan(cloudy_val)] = np.nanmean(cloudy_val)

                        sar_val = sar_raster.read(window=read_wd).astype('float32')
                        sar_val[np.isnan(sar_val)] = np.nanmean(sar_val)

                    data_cloudy = get_normalized_data(cloudy_val,3)
                    data_sar = get_normalized_data(sar_val,1)


                    cloudy_detect = np.transpose(data_cloudy, (1,2,0))
                    sar_detect = np.transpose(data_sar, (1,2,0))

                    cloudy_temp = np.zeros((input_size, input_size, cloudy_detect.shape[2]))
                    sar_temp = np.zeros((input_size, input_size, sar_detect.shape[2]))

                    
                    mask = np.pad(np.ones((stride_size, stride_size), dtype=np.uint8), ((padding, padding),(padding, padding)))
                    shape = (stride_size, stride_size)

                    if y_count < input_size or x_count < input_size:

                        cloudy_temp = np.zeros((input_size, input_size, cloudy_detect.shape[2]))
                        sar_temp = np.zeros((input_size, input_size, sar_detect.shape[2]))

                        mask = np.zeros((input_size, input_size), dtype=np.uint8)
                        if start_x == 0 and start_y == 0:
                            cloudy_temp[(input_size - y_count):input_size, (input_size - x_count):input_size] = cloudy_detect
                            sar_temp[(input_size - y_count):input_size, (input_size - x_count):input_size] = sar_detect

                            mask[(input_size - y_count):input_size-padding, (input_size - x_count):input_size-padding] = 1
                            shape = (y_count-padding, x_count-padding)
                        elif start_x == 0:
                            cloudy_temp[0:y_count, (input_size - x_count):input_size] = cloudy_detect
                            sar_temp[0:y_count, (input_size - x_count):input_size] = sar_detect

                            if y_count == input_size:
                                mask[padding:y_count-padding, (input_size - x_count):input_size-padding] = 1
                                shape = (y_count-2*padding, x_count-padding)
                            else:
                                mask[padding:y_count, (input_size - x_count):input_size-padding] = 1
                                shape = (y_count-padding, x_count-padding)
                        elif start_y == 0:
                            cloudy_temp[(input_size - y_count):input_size, 0:x_count] = cloudy_detect
                            sar_temp[(input_size - y_count):input_size, 0:x_count] = sar_detect

                            if x_count == input_size:
                                mask[(input_size - y_count):input_size-padding, padding:x_count-padding] = 1
                                shape = (y_count-padding, x_count-2*padding)
                            else:
                                mask[(input_size - y_count):input_size-padding, padding:x_count] = 1
                                shape = (y_count-padding, x_count-padding)
                        else:
                            cloudy_temp[0:y_count, 0:x_count] = cloudy_detect
                            sar_temp[0:y_count, 0:x_count] = sar_detect

                            mask[padding:y_count, padding:x_count] = 1
                            shape = (y_count-padding, x_count-padding)
                            
                        cloudy_detect = cloudy_temp
                        sar_detect = sar_temp

                    mask = (mask!=0)

                    if np.count_nonzero(cloudy_detect) > 0:
                        if len(np.unique(cloudy_detect)) <= 2:
                            pass
                        else:

                            cloudy_detect = np.transpose(cloudy_detect, (2,0,1))
                            sar_detect = np.transpose(sar_detect, (2,0,1))
                            
                            cloudy_detect = cloudy_detect[np.newaxis,...]
                            sar_detect = sar_detect[np.newaxis,...]

                            img_pre = [cloudy_detect,sar_detect]
                            y_pred = model.predict(img_pre)
                            y_pred *= 2000

                            y_pred = y_pred[0][0:13]
                            for i in range(y_pred.shape[0]):

                                y = y_pred[i][mask].reshape(shape)
                                with write_lock:
                                    r.write(y, i+1, window=Window(start_x, start_y, shape[1], shape[0]))
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(tqdm(executor.map(process, list_coordinates), total=len(list_coordinates)))

    
if __name__=="__main__":
    cloudy_img_path = '/home/boom/data/cloud_removal/large_images/aoi_s2_cloudy.tif'
    sar_img_path = '/home/boom/Downloads/aoi_s1.tif'
    output_tif = '/home/boom/data/cloud_removal/results/aoi_s2_v2.tif'
    model_path = '/home/boom/boom/dsen2-cr/weights/cloud_removal.h5'

    model = load_model(model_path, compile=False, custom_objects={'tf': tf})
    print(model.summary())
    predict(cloudy_img_path, sar_img_path, output_tif, size=256)