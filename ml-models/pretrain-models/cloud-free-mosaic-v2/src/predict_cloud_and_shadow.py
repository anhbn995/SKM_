import rasterio
import numpy as np


import cv2
from tqdm import *

import logging
from src.model.cloud_shadow_model import build_model

from osgeo import gdal

def predict(img_url, threshold = 0.5, cnn_model=None):
    num_band = 4

    batch_size = 3
    input_size = 512
    INPUT_SIZE = 512
    n_classes = 1

    dataset = gdal.Open(img_url)
    values = dataset.ReadAsArray()[0:num_band].astype(np.float16)
    ## Crop attributes
    h,w = values.shape[1:3]
    crop_size = 512   # input_size//2 # 200
    padding = int((input_size - crop_size)/2)
    padded_org_im = []
    cut_imgs = []
    new_w = w + 2*padding
    new_h = h + 2*padding
    cut_w = list(range(padding, new_w - padding, crop_size))
    cut_h = list(range(padding, new_h - padding, crop_size))

    list_hight = []
    list_weight = []
    for i in cut_h:
        if i < new_h - padding - crop_size:
            list_hight.append(i)
    list_hight.append(new_h-crop_size-padding)

    for i in cut_w:
        if i < new_w - crop_size - padding:
            list_weight.append(i)
    list_weight.append(new_w-crop_size-padding)

    img_coords = []
    for i in list_weight:
        for j in list_hight:
            img_coords.append([i, j])
    
    for i in range(num_band):
        band = np.pad(values[i], padding, mode='reflect')
        padded_org_im.append(band)

    values = np.array(padded_org_im).swapaxes(0,1).swapaxes(1,2)
    del padded_org_im

    def get_im_by_coord(org_im, start_x, start_y,num_band):
        startx = start_x-padding
        endx = start_x+crop_size+padding
        starty = start_y - padding
        endy = start_y+crop_size+padding

        result=[]
        img = org_im[starty:endy, startx:endx]
        img = img.swapaxes(2,1).swapaxes(1,0)
        for chan_i in range(num_band):
            result.append(img[chan_i])
        return np.array(result).swapaxes(0,1).swapaxes(1,2)

    for i in range(len(img_coords)):
        im = get_im_by_coord(
            values, img_coords[i][0], img_coords[i][1],num_band)
        im = im[...,:4]
        # im2 = preprocess_grayscale(im)
        # im2 = normalize_4band_esri(im, num_band)
        # im2 = normalize_255(im)
        # im2 = normalize_bandcut(im, num_band)
        cut_imgs.append(im)
    # print(len(cut_imgs))

    a = list(range(0, len(cut_imgs), batch_size))

    if a[len(a)-1] != len(cut_imgs):
        a[len(a)-1] = len(cut_imgs)
    
    y_pred = []
    for i in tqdm(range(len(a)-1)):
        x_batch = []
        x_batch = np.array(cut_imgs[a[i]:a[i+1]], dtype=np.float16)/np.finfo(np.float16).max
        y_batch = cnn_model.predict(x_batch)

        y_batch = np.array(y_batch, dtype=np.float16)
        y_pred.extend(y_batch)
    big_mask = np.zeros((h, w, 3)).astype(np.float16)
    for i in range(len(cut_imgs)):
        if np.count_nonzero(cut_imgs[i]) > 0:
            true_mask = y_pred[i].reshape((INPUT_SIZE,INPUT_SIZE, 3)).astype(np.float16)
            # true_mask = np.array(true_mask).argmax(axis=2).astype(np.uint8)
            # base_mask = np.zeros((INPUT_SIZE,INPUT_SIZE, 3), dtype=np.uint8)
            # base_mask[...,0] = (true_mask==0)
            # base_mask[...,1] = (true_mask==1)
            # base_mask[...,2] = (true_mask==2)
            # base_mask = base_mask.reshape((INPUT_SIZE,INPUT_SIZE, 3)).astype(np.uint8)
            # true_mask = base_mask # no-overlapped confirm!
        else:
            true_mask = np.zeros((INPUT_SIZE,INPUT_SIZE, 3))
        # true_mask = (cv2.resize(true_mask,(input_size, input_size), interpolation = cv2.INTER_CUBIC)).astype(np.uint8)
        start_x = img_coords[i][1]
        start_y = img_coords[i][0]
        big_mask[start_x-padding:start_x-padding+crop_size, start_y-padding:start_y -
                    padding+crop_size,:] += true_mask[padding:padding+crop_size, padding:padding+crop_size,:]
    del cut_imgs
    label_mask = np.array(big_mask).argmax(axis=2).astype(np.uint8)
    # base_mask = np.zeros((h,w, 3), dtype=np.uint8)
    # base_mask[...,0] = (label_mask==0)
    # base_mask[...,1] = (label_mask==1)
    # base_mask[...,2] = (label_mask==2)
    big_mask = (label_mask).astype(np.uint8)
    # if (big_mask[...,0]+big_mask[...,1]+big_mask[...,2]==np.ones((h, w))).all(): pass
    # else: raise ValueError("Masked overlap !!!")
    return big_mask


def predict_all_2(base_path,outputFileName, model, n_classes = 1, postprocess = False):

    mask_base = predict(base_path, threshold=0.5, cnn_model=model)
    if postprocess:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        dilation = cv2.dilate(mask_base,kernel,iterations = 1)
        erosion = cv2.erode(dilation,kernel,iterations = 1)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
        closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel, iterations = 2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations = 2)
        mask_base = opening
    ###########################################
    with rasterio.open(base_path) as src:
        transform1 = src.transform
        w,h = src.width,src.height
        crs = src.crs
    new_dataset = rasterio.open(outputFileName, 'w', driver='GTiff',
                                height = h, width = w,
                                count=1, dtype="uint8", # change count
                                crs=crs,
                                transform=transform1,
                                compress='lzw')
    new_dataset.write(np.expand_dims(np.array(mask_base), axis=0)) # multiband rasterio's writing order: (bands, cols, rows)  #.swapaxes(0,2).swapaxes(1,2)
    new_dataset.close()

def predict_cloud_shadow(img_data, output_im, weight = None):
    unet = build_model((None,None,4), 46)
    logging.info("Loading model {}".format(unet.name))
    unet.load_weights(weight)
    logging.info("Model loaded !")
    
    predict_all_2(img_data, output_im, model=unet, n_classes = 1, postprocess = True)
    print('Process successfully, output saved at', output_im)
