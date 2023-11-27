# -*- coding: utf-8 -*-
from osgeo import gdal, gdalconst, ogr, osr
import numpy as np
import shapefile as shp
from pyproj import Proj, transform
from math import pi
import glob, os
import sys
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
import time
import skimage.transform
from math import floor
import matplotlib.patches as patches
import argparse
core = multiprocessing.cpu_count()

def cut(img_name, img_dir,box_dir,img_cut_dir):
    source_img =os.path.join(img_dir,img_name+'.tif')    
    img_list_point = get_list_point(os.path.join(box_dir,img_name+'.shp'), source_img)
    print("Image count: " +  str(len(img_list_point)))
    dataset = gdal.Open(source_img)    
    i=0
    all_size = []
    for l in img_list_point:        
        print(l)
        img_cut = img_name+"_cut_"+str(i)+'.tif'
        # img_size = [abs(l[2][0]-l[0][0]), abs(l[2][1]-l[0][1])]
        min_x_img, min_y_img = list(np.amin(l, axis=0))
        max_x_img, max_y_img = list(np.amax(l, axis=0))
        img_size = [abs(max_x_img-min_x_img), abs(max_y_img-min_y_img)]
        all_size.append(img_size)
        print(min_x_img, min_y_img, img_size[0], img_size[1])
        gdal.Translate(os.path.join(img_cut_dir,img_cut), dataset, srcWin = [min_x_img, min_y_img, img_size[0], img_size[1]])        
        i=i+1
    return True


def cut_ones_image(fp_img, fp_shp, out_dir):  
    img_list_point = get_list_point(fp_shp, fp_img)
    print("Image count: " +  str(len(img_list_point)))
    dataset = gdal.Open(fp_img)    
    i=0
    all_size = []
    img_name = os.path.basename(fp_img)
    for l in img_list_point:        
        print(l)
        img_cut = img_name.replace('.tif', "_cut_"+str(i)+'.tif')
        min_x_img, min_y_img = list(np.amin(l, axis=0))
        max_x_img, max_y_img = list(np.amax(l, axis=0))
        img_size = [abs(max_x_img-min_x_img), abs(max_y_img-min_y_img)]
        all_size.append(img_size)
        print(min_x_img, min_y_img, img_size[0], img_size[1])
        gdal.Translate(os.path.join(out_dir, img_cut), dataset, srcWin = [min_x_img, min_y_img, img_size[0], img_size[1]])        
        i=i+1
    return True


def box_not_exist(box, arr_box):    
    return (box not in arr_box)

def box1_is_inside_box2(box1, box2):    
    min_x_b1, min_y_b1 = list(np.amin(box1, axis=0))
    max_x_b1, max_y_b1 = list(np.amax(box1, axis=0))    

    min_x_b2, min_y_b2 = list(np.amin(box2, axis=0))
    max_x_b2, max_y_b2 = list(np.amax(box2, axis=0))

    return min_x_b1 >= min_x_b2 and max_x_b1 <= max_x_b2 and min_y_b1 >= min_y_b2 and max_y_b1 <= max_y_b2

def list_array_to_contour(list_array):
    contour = np.asarray(list_array,dtype=np.float32)
    contour_rs = contour.reshape(len(contour),1,2)
    return contour_rs

def list_list_array_to_list_contour(list_list_array):
    list_contour = []
    for list_array in list_list_array:
        contour = list_array_to_contour(list_array)
        list_contour.append(contour)
    return list_contour

def get_list_point(img_shape_path,source_img):
    print(img_shape_path)
    sf=shp.Reader(img_shape_path)
    my_shapes=sf.shapes()
    my_shapes_list = list(map(lambda shape: shape.points, my_shapes))
    clean_shape_list = []
    for i in range(len(my_shapes_list)):
        if len(my_shapes_list[i]) != 0 and box_not_exist(my_shapes_list[i],clean_shape_list):
            clean_shape_list.append(my_shapes_list[i])

    #doc toa do shape
    driverName = "ESRI Shapefile"
    driver = ogr.GetDriverByName(driverName)
    dataSource = driver.Open(img_shape_path, 0)
    layer = dataSource.GetLayer()
    crs = layer.GetSpatialRef()
    print(crs)
    epsr1 =  crs.GetAttrValue('AUTHORITY',1)


    #doc anh
    driver = gdal.GetDriverByName('GTiff')
    dataset = gdal.Open(source_img)
    proj = osr.SpatialReference(wkt=dataset.GetProjection())
    epsr2 = (proj.GetAttrValue('AUTHORITY',1))    

    #cat chuyen toa do
    # epsr1 = 32750d
    # epsr2 = 4326
    inProj = Proj(init='epsg:%s'%(epsr1))
    outProj = Proj(init='epsg:%s'%(epsr2))

    list_list_point_convert = []
    for shapes in clean_shape_list:
        list_point=[]
        for point in shapes:
            long,lat = point[0],point[1]
            x,y = transform(inProj,outProj,long,lat)
            list_point.append((x,y))
        list_list_point_convert.append(list_point)

    "chuyen sang toa do pixel"
    transformmer = dataset.GetGeoTransform()

    xOrigin = transformmer[0]
    yOrigin = transformmer[3]
    pixelWidth = transformmer[1]
    pixelHeight = -transformmer[5]

    list_list_point=[]
    for points_list in list_list_point_convert :    
        lis_poly=[]
        for point in points_list:
            col = int((point[0] - xOrigin) / pixelWidth)
            row = int((yOrigin - point[1] ) / pixelHeight)
            lis_poly.append([col,row])
        lis_poly = np.asarray(lis_poly,dtype = np.int)
        list_list_point.append(lis_poly)

    del dataset
    return list_list_point

def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.tif"):
        list_id.append(file[:-4])
    return list_id

def main(img_path,box_path):    
    img_list = create_list_id(img_path)
    parent = os.path.dirname(img_path)
    # parent = os.path.dirname(parent)
    foder_name = os.path.basename(img_path)    
    img_cut_dir = os.path.join(parent,"Data_Train_and_Model",foder_name+'_cut_img')
    if not os.path.exists(img_cut_dir):
        os.makedirs(img_cut_dir)        
    p_cnt = Pool(processes=core)    
    result = p_cnt.map(partial(cut,img_dir=img_path,box_dir=box_path,img_cut_dir=img_cut_dir), img_list)
    p_cnt.close()
    p_cnt.join()    
    return img_cut_dir


if __name__ == "__main__":
    """
        RUN WITH 2 IMAGE
    """
    # fp_img = r"/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Full_Images_LandslideDetection/S2B_L2A_20190221_Full.tif"
    # fp_shp = r"/home/skm/SKM16/X/Lo/VungKhongTruotLo/VungKhongTruot.shp"
    # out_dir = r"/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/bo_them_nodata"
    # os.makedirs(out_dir, exist_ok=True)
    # cut_ones_image(fp_img, fp_shp, out_dir)
    
    """
        RUN WITH FOLDER
    """
    # dir_img = os.path.abspath(sys.argv[1])
    # dir_box = os.path.abspath(sys.argv[2])
    # main(dir_img, dir_box)
    # dir_box = r"/home/skm/SKM16/A_CAOHOC/ALL_DATA/label/box/box_Grass"
    # dir_img = r'/home/skm/SKM16/A_CAOHOC/ALL_DATA/img_unit8/RS_OKE/RS_UNET'
    # out_dir = r"/home/skm/SKM16/A_CAOHOC/ALL_DATA/img_unit8/RS_OKE/RS_UNET_cut"
    
    
    # dir_box = r"/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/Labels/label_Green_cover/label_mosaic/box"
    # dir_img = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori'
    # out_dir = r"/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/training_with_original/green_trainig/img_ori_cut"
    
    
    dir_box = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/b_LabelBox_each_class_and_area/Green/B/box_V2"
    dir_img = r'/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/a_img_original/B'
    out_dir = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/DataTraining_Origin/Green/img_ori_cut_V2"
    
    os.makedirs(out_dir, exist_ok=True)

    list_fp_have_box = glob.glob(os.path.join(dir_box, "*.shp"))
    for fp_box in list_fp_have_box:
        fn = os.path.basename(fp_box).replace(".shp","")
        fp_img = os.path.join(dir_img, fn +  ".tif")
        print(fp_img)
        print(fp_box)
        cut_ones_image(fp_img, fp_box, out_dir)
        
        
