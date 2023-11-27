import rasterio
import numpy as np
import copy
import skimage
import cv2

def get_water_mask(multiclass_path):
    with rasterio.open(multiclass_path) as src:
        img_all = src.read()

    water_img = copy.deepcopy(img_all)
    water_img[water_img!=4]=0
    water_img[water_img==4]=1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask_water = water_img[0].astype(bool)
    mask_water = skimage.morphology.binary_dilation(mask_water, kernel)
    mask_water = skimage.morphology.remove_small_objects(mask_water, min_size=50000)
    return mask_water

def get_mangrove_mask(mangrove_path):
    with rasterio.open(mangrove_path) as src:
        mangrove_data = 1 - src.read()

    mangrove_img = copy.deepcopy(mangrove_data)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask_mangrove = mangrove_img[0].astype(bool)
    mask_mangrove = skimage.morphology.binary_dilation(mask_mangrove, kernel)
    mask_mangrove = skimage.morphology.remove_small_objects(mask_mangrove, min_size=30000)
    return mask_mangrove

def get_green_mask(multiclass_path):
    with rasterio.open(multiclass_path) as src:
        img_all = src.read()

    green_img = copy.deepcopy(img_all)
    green_img[green_img==1]=2
    green_img[green_img==3]=2
    green_img[green_img==2]=1
    green_img[green_img==4]=0
    green_img = green_img[0].astype(bool)
    mask_green = skimage.morphology.remove_small_holes(green_img, area_threshold=10000)
    return mask_green

def get_grass_mask(grass_built_path):
    with rasterio.open(grass_built_path) as src:
        grass_build_data = src.read()

    grass_img = copy.deepcopy(grass_build_data)
    grass_img[grass_img!=2]=0
    grass_img[grass_img==2]=1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask_grass = grass_img[0].astype(bool)
    mask_grass = skimage.morphology.binary_dilation(mask_grass, kernel)
    mask_grass = skimage.morphology.remove_small_objects(mask_grass, min_size=500)
    return mask_grass

def get_builtup_mask(grass_built_path):
    with rasterio.open(grass_built_path) as src:
        grass_build_data = src.read()

    builtup_img = copy.deepcopy(grass_build_data)
    builtup_img[builtup_img!=3]=0
    builtup_img[builtup_img==3]=1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_builtup = builtup_img[0].astype(bool)
    mask_builtup = skimage.morphology.binary_dilation(mask_builtup, kernel)
    mask_builtup = skimage.morphology.remove_small_objects(mask_builtup, min_size=50)
    return mask_builtup
    
def gen_final_result(origin_img, mask_green, mask_grass, mask_mangrove, mask_water, mask_builtup):
    with rasterio.open(origin_img) as src:
        height = src.height
        width = src.width

    final_img = np.zeros((height,width))
    final_img[mask_green==True] = 2
    final_img[mask_grass==True] = 5

    final_img[mask_mangrove==True] = 1
    final_img[mask_water==True] = 4
    final_img[mask_builtup==True] = 3
    return final_img
