import tqdm
import glob, os
import json, rasterio
import sys
import geopandas as gp
import numpy as np
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
import datetime
import cv2, math
from tqdm import *
import pandas as pd
core = multiprocessing.cpu_count()
def multi2single(gpdf):
    gpdf_singlepoly = gpdf[gpdf.geometry.type == 'Polygon']
    gpdf_multipoly = gpdf[gpdf.geometry.type == 'MultiPolygon']

    for i, row in gpdf_multipoly.iterrows():
        Series_geometries = pd.Series(row.geometry)
        df = pd.concat([gp.GeoDataFrame(row, crs=gpdf_multipoly.crs).T]*len(Series_geometries), ignore_index=True)
        df['geometry']  = Series_geometries
        gpdf_singlepoly = pd.concat([gpdf_singlepoly, df])
    gpdf_singlepoly.reset_index(inplace=True, drop=True)
    return gpdf_singlepoly
def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.tif"):
        list_id.append(file[:-4])
        # print(file[:-4])
    return list_id

def convert_geography_to_pixel(x):
    _segmentation = []
    for a in x.exterior.coords:
        x = float(int((a[0] - transform[2])/ transform[0]))
        y = float(int((a[1] - transform[5])/ transform[4]))
        _segmentation.extend((x,y))
    return _segmentation

def areaes(x):
    return round((x.area/transform[0]/-transform[4]),2)

def boxes(x):
    _bbox = [float(int((x.bounds[3] - transform[5])/ transform[4])),
            float(int((x.bounds[0] - transform[2])/ transform[0])),
            float(math.ceil((x.bounds[1] - x.bounds[3])/ transform[4])),
            float(math.ceil((x.bounds[2] - x.bounds[0])/ transform[0]))]
    return _bbox
    
def create_list_annotation(image_name,shape_dir,image_dir):
    
    with rasterio.open(os.path.join(image_dir, image_name+".tif")) as r:
        global transform
        transform = r.transform
        height, width = r.height, r.width
        projstr = r.crs.to_string()

    # try:
    if os.path.exists(os.path.join(shape_dir, image_name+'.shp')):
        shapefile = gp.read_file(os.path.join(shape_dir, image_name+'.shp'))
        shapefile = multi2single(shapefile)
        # crs = {'init':'epsg:3857'}
        # shapefile = shapefile.to_crs(crs)
        # shapefile['area'] = shapefile.area
        # shapefile = shapefile[shapefile['area'] > 10]
        # shapefile = shapefile.to_crs(projstr)
        # print(shapefile.type)
        shapefile = shapefile[shapefile.geometry != None]
        shapefile = shapefile.reset_index()
        # list_id = list(range(len(shapefile)))
        # shapefile["id"]
        # print(shapefile)
        # shapefile = shapefile.to_crs(projstr)
        

        list_cnt = shapefile["geometry"].apply(convert_geography_to_pixel)
        area = shapefile["geometry"].apply(areaes)
        box = shapefile["geometry"].apply(boxes)
        list_cnt_ressult = [list_cnt[i] for i in range(len(list_cnt)) if area[i]>30]
        list_area_ressult = [area[i] for i in range(len(list_cnt)) if area[i]>30]
        list_box_ressult = [box[i] for i in range(len(list_cnt)) if area[i]>30]
    else:
        list_cnt_ressult=[]
        list_area_ressult=[]
        list_box_ressult=[]
    # except:
    #     list_cnt=[]
    #     area=[]
    #     box=[]
    return height, width, list_cnt_ressult, list_area_ressult, list_box_ressult

def main_gen_anotation(image_dir,shape_dir):
    parent = os.path.dirname(image_dir)
    _final_object = {}
    _final_object["info"]= {
                            "contributor": "crowdAI.org",
                            "about": "Dataset for crowdAI Mapping Challenge",
                            "date_created": datetime.datetime.utcnow().isoformat(' '),
                            "description": "crowdAI mapping-challenge dataset",
                            "url": "https://www.crowdai.org/challenges/mapping-challenge",
                            "version": "1.0",
                            "year": 2018
                            }
                        
    _final_object["categories"]=[
                    {
                        "id": 100,
                        "name": "building",
                        "supercategory": "building"
                    }
                ]
    date_captured=datetime.datetime.utcnow().isoformat(' ')
    license_id=1
    coco_url=""
    flickr_url=""
    _images = []
    _annotations = []
    _list_image = create_list_id(image_dir)
    _image_id = 0
    _annotation_id = 0
    with tqdm(total=len(_list_image)) as pbar:
        for _image_name in _list_image:
            pbar.update()
            if os.path.exists(os.path.join(shape_dir, _image_name+'.shp')) and os.path.exists(os.path.join(image_dir, _image_name+'.tif')):
                # print(111)
                _image_id = _image_id + 1
                _file_name = _image_name + '.tif'
                _height, _width, list_segmentation, list_area, list_box = create_list_annotation(_image_name,shape_dir,image_dir)
                if len(list_box)>0:
                    image_info = {
                                "id": _image_id,
                                "file_name": _file_name,
                                "width": _width,
                                "height": _height,
                                "date_captured": date_captured,
                                "license": license_id,
                                "coco_url": coco_url,
                                "flickr_url": flickr_url
                                }
                    _images.append(image_info)
                    for idxxx, _segmentation in enumerate(list_segmentation):
                        _annotation_id = _annotation_id + 1
                        annotation_info = {
                            "id": _annotation_id,
                            "image_id": _image_id,
                            "category_id": 100,
                            "iscrowd": 0,
                            "area": list_area[idxxx],
                            "bbox": list_box[idxxx],
                            "segmentation": _segmentation,
                            "width": _width,
                            "height": _height,
                        }
                        _annotations.append(annotation_info)
    _final_object["images"]=_images
    _final_object["annotations"]=_annotations
    fp = open(os.path.join(parent,'annotation.json'), "w")
    print("Writing JSON...")
    fp.write(json.dumps(_final_object))
    print("Done!")

if __name__ == "__main__":
    image_dir = "/media/skymap/Data/Building_master_data/data_result/val/images"
    shape_dir = "/media/skymap/Data/Building_master_data/data_result/val/images_shape/"
    main_gen_anotation(image_dir,shape_dir)
