import glob, os
from osgeo import gdal, gdalconst, ogr, osr
import numpy as np
import math
import sys
from pyproj import Proj, transform

def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.tif"):
        list_id.append(file[:-4])
        # print(file[:-4])
    return list_id

def main_split(foder_image,split,out_dir_data):
    import shutil

    image_list = create_list_id(foder_image)
    np.random.shuffle(image_list)
    count = len(image_list)
    cut_idx = int(round(count*split))
    print(cut_idx)
    train_list = image_list[0:cut_idx]
     
    # val_list = image_list[cut_idx:count]
    other_list = [id_image for id_image in image_list if id_image not in train_list]
#     cut_2 = len(other_list)//2
#     val_list = other_list[0:cut_2]
#     test_list = other_list[cut_2:]
    val_list = other_list
    test_list = []
    path_train = os.path.join(out_dir_data,'train','images')
    path_train_mask = os.path.join(out_dir_data,'train','mask')
    if not os.path.exists(path_train):
        os.makedirs(path_train)
    if not os.path.exists(path_train_mask):
        os.makedirs(path_train_mask)
    path_val = os.path.join(out_dir_data,'val','images')
    path_val_mask = os.path.join(out_dir_data,'val','mask')
    if not os.path.exists(path_val):
        os.makedirs(path_val)
    if not os.path.exists(path_val_mask):
        os.makedirs(path_val_mask)
    path_test = os.path.join(out_dir_data,'test','images')
    path_test_mask = os.path.join(out_dir_data,'test','mask')
    if not os.path.exists(path_test):
        os.makedirs(path_test)
    if not os.path.exists(path_test_mask):
        os.makedirs(path_test_mask)
    for image_name in train_list:
        # shutil.copy(os.path.join(foder_image,image_name+'.tif'), path_train)
        os.rename(os.path.join(foder_image,image_name+'.tif'), os.path.join(path_train,image_name+'.tif'))
        # os.rename(os.path.join(foder_image_mask,image_name+'.tif'), os.path.join(path_train_mask,image_name+'.tif'))
    for image_name in val_list:
        # shutil.copy(os.path.join(foder_image,image_name+'.tif'), path_val)
        os.rename(os.path.join(foder_image,image_name+'.tif'), os.path.join(path_val,image_name+'.tif'))
        # os.rename(os.path.join(foder_image_mask,image_name+'.tif'), os.path.join(path_val_mask,image_name+'.tif'))
    for image_name in test_list:
        # shutil.copy(os.path.join(foder_image,image_name+'.tif'), path_val)
        os.rename(os.path.join(foder_image,image_name+'.tif'), os.path.join(path_test,image_name+'.tif'))
        # os.rename(os.path.join(foder_image_mask,image_name+'.tif'), os.path.join(path_test_mask,image_name+'.tif'))
    return path_train,path_val,path_test
    # return path_train,path_val
if __name__=='__main__':
    foder_image = os.path.abspath(sys.argv[1])
# foder_image_mask = os.path.abspath(sys.argv[2])

    split = float(sys.argv[2])
    out_dir_data = os.path.abspath(sys.argv[3])
    main_split(foder_image,split,out_dir_data)
