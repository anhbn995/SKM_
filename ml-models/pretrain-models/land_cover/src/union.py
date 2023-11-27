import numpy as np
import rasterio
from rasterio.windows import Window
import threading
from tqdm import tqdm
import concurrent.futures
import warnings, cv2, os
import tensorflow as tf
# import Vectorization
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects
from utils import get_range_value, create_list_id
from tensorflow.compat.v1.keras.backend import set_session
from model import unet_basic
import glob
from params import *
from config import *


warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))


def predict(model, path_image, path_predict, size=128):
    print(path_image)
    # qt_scheme = get_quantile_schema(path_image)
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
        with rasterio.open(path_predict,'w+', **meta, compress='lzw') as r:
            read_lock = threading.Lock()
            write_lock = threading.Lock()

            def process(coordinates):
                x_off, y_off, x_count, y_count, start_x, start_y = coordinates
                read_wd = Window(x_off, y_off, x_count, y_count)
                with read_lock:
                    values = raster.read(window=read_wd)
                if raster.profile["dtype"]=="uint8":
                    image_detect = values[0:3].transpose(1,2,0).astype(int)
                else:
                    # datas = []
                    # for chain_i in range(4):
                    #     # band_qt = qt_scheme[chain_i]
                    #     band = values[chain_i]

                    #     # cut_nor = np.interp(band, (band_qt.get('p2'), band_qt.get('p98')), (1, 255)).astype(int)
                    #     cut_nor = get_range_value(band)
                    #     datas.append(cut_nor)
                    datas = get_range_value(values)
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
                        y_pred = model.predict(image_detect[np.newaxis,...])
                        y_pred = (y_pred[0,...,0] > 0.5).astype(np.uint8) 
                        
                        # y_pred = 1 - y_pred
                        y = y_pred[mask].reshape(shape)
                        
                        with write_lock:
                            r.write(y[np.newaxis,...], window=Window(start_x, start_y, shape[1], shape[0]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                results = list(tqdm(executor.map(process, list_coordinates), total=len(list_coordinates)))
                

def gop_anh(list_fn_image_origin, thu_tu_lop, dir_predict_all, dir_out_union):
    os.makedirs(dir_out_union, exist_ok=True)
    in_first_class = thu_tu_lop[0][:-1]
    value_first = int(thu_tu_lop[0][-1])
    
    for fn_img in list_fn_image_origin:
        fp_out_union = os.path.join(dir_out_union, fn_img + '.tif')
        first_fp_class = os.path.join(dir_predict_all, fn_img, fn_img+ f'_{in_first_class}.tif')
        with rasterio.open(first_fp_class) as src:
            img = src.read()
            meta = src.meta
        img[img!=0] = value_first
        for lop_value in thu_tu_lop[1:]:
            lop_ = lop_value[:-1]
            value_ = int(lop_value[-1])
            
            fp_ = os.path.join(dir_predict_all, fn_img, fn_img+ f'_{lop_}.tif')
            with rasterio.open(fp_) as src:
                img_ = src.read()
                idex_khac0 = np.where(img_ != 0)
                img[idex_khac0] = value_
        with rasterio.open(fp_out_union, 'w', **meta) as dst:
            dst.write(img)
                

def main(dict_model_path, thu_tu_lop, folder_image_path, dir_predict_all, dir_out_union, size_model=256):
    model = unet_basic((size_model, size_model, 4))
    
    if not os.path.exists(dir_predict_all):
        os.makedirs(dir_predict_all)
        
    list_fn_image_origin = create_list_id(folder_image_path)
    print('Start predict ... ')
    for class_name in thu_tu_lop:
        fp_model = dict_model_path[class_name[:-1]]
        model.load_weights(fp_model)
        
        for fname in tqdm(list_fn_image_origin) :
            dir_out_1_image = os.path.join(dir_predict_all, fname)
            
            os.makedirs(dir_out_1_image, exist_ok=True)
            fp_img = os.path.join(folder_image_path, fname + '.tif')
            fp_predict = os.path.join(dir_out_1_image, fname + f'_{class_name[:-1]}.tif')
            predict(model, fp_img, fp_predict, size_model)
    print('Done predict!')
    print('Merge value ... ')
    gop_anh(list_fn_image_origin, thu_tu_lop, dir_predict_all, dir_out_union)
    

def gop_anh_single(fn_img, thu_tu_lop, dir_predict_all, fp_out_union):
    in_first_class = thu_tu_lop[0][:-1]
    value_first = int(thu_tu_lop[0][-1])
    
    first_fp_class = os.path.join(dir_predict_all, fn_img, fn_img+ f'_{in_first_class}.tif')
    with rasterio.open(first_fp_class) as src:
        img = src.read()
        meta = src.meta
    img[img!=0] = value_first
    for lop_value in thu_tu_lop[1:]:
        lop_ = lop_value[:-1]
        value_ = int(lop_value[-1])
        
        fp_ = os.path.join(dir_predict_all, fn_img, fn_img+ f'_{lop_}.tif')
        with rasterio.open(fp_) as src:
            img_ = src.read()
            idex_khac0 = np.where(img_ != 0)
            img[idex_khac0] = value_
    with rasterio.open(fp_out_union, 'w', **meta) as dst:
        dst.write(img)


def main_single(dict_model_path, thu_tu_lop, fp_image, dir_predict_all, fp_out_union, size_model=256):
    model = unet_basic((size_model, size_model, 4))
    
    
    if not os.path.exists(dir_predict_all):
        os.makedirs(dir_predict_all)
    fname = os.path.basename(fp_image)[:-4]
    dir_out_1_image = os.path.join(dir_predict_all, fname)
    os.makedirs(dir_out_1_image, exist_ok=True)
    for class_name in thu_tu_lop:
        fp_model = dict_model_path[class_name[:-1]]
        model.load_weights(fp_model)
        
        fp_predict = os.path.join(dir_out_1_image, fname + f'_{class_name[:-1]}.tif')
        predict(model, fp_image, fp_predict, size_model)
    print('Done predict!')
    print('Merge value ... ')
    gop_anh_single(fname, thu_tu_lop, dir_predict_all, fp_out_union)

if __name__=='__main__':
    
    # DICT_MODEL_PATH = {
    #     "buildUp": f'{ROOT_DATA_FOLDER}/pretrain-models/land_cover/v1/weights/model_building256_2.h5',
    #     "green": f'{ROOT_DATA_FOLDER}/pretrain-models/land_cover/v1/weights/model_green_last_v3.h5',
    #     "water": f'{ROOT_DATA_FOLDER}/pretrain-models/land_cover/v1/weights/model_water_v5.h5',
    #     "vacant": f'{ROOT_DATA_FOLDER}/pretrain-models/land_cover/v1/weights/model_dat_trong_v3.h5'
    # } 
    
        
    DICT_MODEL_PATH = {
        "buildUp": r'/home/skm/SKM16/Tmp/LAND COVER/HieuGREEN/weights_ok/model_building256_2.h5',
        "green": r'/home/skm/SKM16/Tmp/LAND COVER/HieuGREEN/weights_ok/model_green_last_v3.h5',
        "water": r'/home/skm/SKM16/Tmp/LAND COVER/HieuGREEN/weights_ok/model_water_v5.h5',
        "vacant": r'/home/skm/SKM16/Tmp/LAND COVER/HieuGREEN/weights_ok/model_dat_trong_v3.h5'
    } 
    
    # INPUT_PATH = r'/home/skm/public_mount/tmp_ducanh/LANDCOVER/predict/test/test2/berlin2020.tif'
    # TMP_PATH = r'/home/skm/SKM16/Tmp/LAND COVER/HieuGREEN/hieu_test/smar'
    # THU_TU_LOP = ['vacant1','green2','water3','buildUp4']
    # OUTPUT_PATH = r'/home/skm/SKM16/Tmp/LAND COVER/HieuGREEN/hieu_test/mar1/hieu_bo.tif'
    # os.makedirs('/home/skm/SKM16/Tmp/LAND COVER/HieuGREEN/hieu_test/mar1', exist_ok=True)
    main_single(DICT_MODEL_PATH, THU_TU_LOP, INPUT_PATH, TMP_PATH, OUTPUT_PATH, SIZE_MODEL)
    
    

    
    # folder_image_path = r'/home/skm/public_mount/tmp_ducanh/LANDCOVER/predict/test'
    # dir_predict_all = r'/home/skm/SKM16/Tmp/LAND COVER/HieuGREEN/hieu_test/a112'
    # thu_tu_lop = ['vacant1','green2','water3','buildUp4']
    # dir_out_union = r'/home/skm/SKM16/Tmp/LAND COVER/HieuGREEN/hieu_test/a1123'
  
    # main(DICT_MODEL_PATH, THU_TU_LOP, FOLDER_IMAGE_PATH, DIR_PREDICT_ALL, DIR_OUT_UNION, SIZE_MODEL)
