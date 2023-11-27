# -*- coding: utf-8 -*-
from osgeo import gdal, gdalconst, ogr, osr
import numpy as np
import glob, os
import sys
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
import time
import argparse
import rasterio
import rasterio.mask
import geopandas as gp

core = multiprocessing.cpu_count()//4

def cut(img_name, img_dir,box_dir,img_cut_dir):
    image_path = os.path.join(img_dir,img_name+'.tif')
    shape_path = os.path.join(box_dir,img_name+'.shp')

    with rasterio.open(image_path, mode='r+') as src:
        projstr = src.crs.to_string()
        print(projstr)
        check_epsg = src.crs.is_epsg_code
        if check_epsg:
            epsg_code = src.crs.to_epsg()
            print(epsg_code)
        else:
            epsg_code = None
    if epsg_code:
        out_crs = {'init':'epsg:{}'.format(epsg_code)}
    else:
        out_crs = projstr
    bound_shp = gp.read_file(shape_path)
    bound_shp = bound_shp.to_crs(out_crs)

    for index2, row_bound in bound_shp.iterrows():
        geoms = row_bound.geometry
        img_cut = img_name+"_{}.tif".format(index2)
        img_cut_path = os.path.join(img_cut_dir,img_cut)
        try:
            with rasterio.open(image_path,BIGTIFF='YES') as src:
                out_image, out_transform = rasterio.mask.mask(src, [geoms], crop=True)
                out_meta = src.meta
            out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})
            with rasterio.open(img_cut_path, "w", **out_meta) as dest:
                dest.write(out_image)
        except Exception:
            print(img_name,index2)

    return True
    
def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.tif"):
        list_id.append(file[:-4])
    return list_id

def main_cut_img(img_path,box_path,out_dir):    
    img_list = create_list_id(img_path)
    parent = os.path.dirname(img_path)
    # parent = os.path.dirname(parent)
    foder_name = os.path.basename(img_path)    
    img_cut_dir = os.path.join(out_dir,"tmp",foder_name+'_cut_img')
    if not os.path.exists(img_cut_dir):
        os.makedirs(img_cut_dir)    
    for image_name in img_list:
        cut(image_name,img_path,box_path,img_cut_dir)
    # p_cnt = Pool(processes=core)    
    # result = p_cnt.map(partial(cut,img_dir=img_path,box_dir=box_path,img_cut_dir=img_cut_dir), img_list)
    # p_cnt.close()
    # p_cnt.join()    
    return img_cut_dir


if __name__ == "__main__":
    # args_parser = argparse.ArgumentParser()

    # args_parser.add_argument(
    #     '--image_dir',
    #     help='Orginal Image Directory',
    #     required=True
    # )


    # args_parser.add_argument(
    #     '--box_dir',
    #     help='Box cut directory',
    #     required=True
    # )

    # param = args_parser.parse_args()
    
    # img_path = str(param.image_dir)
    # box_path = str(param.box_dir)
    img_path=r"/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien"
    box_path=r"/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/ShipDetection/box"
    out_path=r"/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/data_faster-rnn/image_crop_box"
    main_cut_img(img_path,box_path,out_path)    


