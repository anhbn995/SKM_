from curses import meta
import rasterio
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import os
import cv2
from rasterio.windows import Window

def write_window_many_chanel(output_ds, arr_c, s_h, e_h ,s_w, e_w, sw_w, sw_h, size_w_crop, size_h_crop):
    for c, arr in enumerate(arr_c):
        output_ds.write(arr[s_h:e_h,s_w:e_w],window = Window(sw_w, sw_h, size_w_crop, size_h_crop), indexes= c + 1)

def get_df_flatten_predict(img_window, name_atrr):
    dfObj = pd.DataFrame()
    i = 0
    for band in img_window:
        band = band.flatten()
        name_band = f"band {i + 1}_{name_atrr}"
        dfObj[name_band] = band
        i+=1
    return dfObj

def create_df_from_window_oke(fp_img_stack, w_crop_start, h_crop_start, size_w_crop, size_h_crop):
    with rasterio.open(fp_img_stack) as src:
        img_window = src.read(window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
    df_all_predict = get_df_flatten_predict(img_window, 'band')  
    return df_all_predict, img_window.shape[1:]

def add_building(result_nobuilding_tif, result_building_tif):
    src_img = rasterio.open(result_nobuilding_tif)
    src_buid = rasterio.open(result_building_tif)

    index_build = np.where(src_buid.read()==255)
    mask = src_img.read() + 1
    index_2 = np.where(mask == 2)
    mask[index_2] = 1
    mask[index_build] = 2
    mask = mask - 1

    with rasterio.open(result_nobuilding_tif, "r+") as dst:
        dst.write(mask)
        dst.write_colormap(
                        1, {
                            1: (237,2,42,255),
                            2: (255,219,92,255),
                            3:(167,210,130,255),
                            4:(200,200,200,255),
                            5:(238,207,168,255),
                            6:(53,130,33,255),
                            7:(26,91,171,255)
                            })

def morpho_result(pre_morpho, final_result):
    with rasterio.open(pre_morpho) as src:
        predict = src.read(1)
        meta = src.meta.copy()
        mask_rest = (src.read_masks(1)).astype(np.uint8)

    class_value_sort = [1,6,3,2,7,5,4]

    list_mask =[]
    kernel = np.ones((3,3),np.uint8)
    for value in class_value_sort[0:]:
        mask_class = ((predict == value)*255).astype(np.uint8)
        mask_class = cv2.bitwise_and(mask_rest, mask_class)
        mask_class = cv2.morphologyEx(mask_class, cv2.MORPH_CLOSE, kernel)
        mask_class = cv2.morphologyEx(mask_class, cv2.MORPH_OPEN, kernel)
        
        mask_class = cv2.bitwise_and(mask_rest, mask_class)
        mask_rest = mask_rest - mask_class

        list_mask.append(mask_class)
    list_mask[-1]= cv2.bitwise_or(mask_class,mask_rest)
    mophol_result = np.zeros(predict.shape).astype(np.uint8)

    list_pixel = []
    for i in range(len(class_value_sort)):
        value = class_value_sort[i]
        mask_i = list_mask[i]==255
        list_pixel.append(np.count_nonzero(mask_i))
        mophol_result[mask_i] = value

    with rasterio.open(final_result,'w', **meta) as dst:
        dst.write(mophol_result, indexes=1)
        dst.write_colormap(
                        1, {
                            1: (237,2,42,255),
                            2: (255,219,92,255),
                            3:(167,210,130,255),
                            4:(200,200,200,255),
                            5:(238,207,168,255),
                            6:(53,130,33,255),
                            7:(26,91,171,255)
                            })
        return True


        return True