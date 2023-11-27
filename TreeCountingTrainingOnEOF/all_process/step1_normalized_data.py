import numpy as np
import rasterio
import rasterio.features
import geopandas as gp
import pandas as pd
from shapely.geometry import shape, mapping
from matplotlib import pyplot as plt
import time
from shapely import geometry
import os,glob
def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.tif"):
        list_id.append(file[:-4])
    return list_id
def read_label(shape_tree_path,mode=0):
    crs = {'init':'epsg:3857'}
    classify_shp = gp.read_file(shape_tree_path)
    classify_shp = classify_shp.to_crs(crs)
    list_point = []
    if mode == 0:
        for index, geo_row in classify_shp.iterrows():
            geom = geo_row.geometry
            list_point.append((geometry.shape(geom).centroid))
    elif mode == 1:
        for index, geo_row in classify_shp.iterrows():
            geom = geo_row.geometry
            list_point.append((geometry.shape(geom)))
    return list_point

def buffer_point_to_polygon(list_point,distance = 0.4):
    list_geometry = [point.buffer(distance) for point in list_point]
    return list_geometry

def std_shape(shape_path,image_path,output_path,mode = 1):
    if mode == 0:
        list_point = read_label(shape_path,mode = mode)
        list_geometry = buffer_point_to_polygon(list_point)
    else:
        list_geometry = read_label(shape_path,mode = 1)

    data_fame = pd.DataFrame(list_geometry, columns=['geometry'])
    crs = {'init':'epsg:3857'}
    gdf = gp.GeoDataFrame(data_fame, geometry='geometry',crs=crs)
    print(gdf, "1111")
    if mode == 2:
        gdf['area'] = gdf.area
        gdf = gdf[gdf['area'] < 900]
    print(gdf,"2222")
    with rasterio.open(image_path, mode='r+') as src:
        # projstr = (src.crs.to_proj4())
        projstr = src.crs.to_string()
#         print(projstr)
        check_epsg = src.crs.is_epsg_code
        if check_epsg:
            epsg_code = src.crs.to_epsg()
#             print(epsg_code)
        else:
            epsg_code = None
    if epsg_code:
        out_crs = {'init':'epsg:{}'.format(epsg_code)}
    else:
        out_crs = projstr
    gdf = gdf.to_crs(out_crs)
    gdf.to_file(output_path)

def main_std(image_dir,box_dir,label_dir,out_dir):
    list_id = create_list_id(image_dir)
    print(list_id)
    box_out_dir = os.path.join(out_dir,"box_std")
    label_out_dir = os.path.join(out_dir,"label_std")
    if not os.path.exists(box_out_dir):
        os.makedirs(box_out_dir) 
    if not os.path.exists(label_out_dir):
        os.makedirs(label_out_dir) 
    for image_id in list_id:
        image_path = os.path.join(image_dir,image_id+'.tif')
        label_path = os.path.join(label_dir,image_id+'.shp')
        box_path = os.path.join(box_dir,image_id+'.shp')
        out_label_path = os.path.join(label_out_dir,image_id+'.shp')
        out_box_path = os.path.join(box_out_dir,image_id+'.shp')
        print(image_id)
        std_shape(label_path,image_path,out_label_path,mode = 2)
        std_shape(box_path,image_path,out_box_path,mode = 1)
    return box_out_dir, label_out_dir

if __name__ == '__main__':
    main_std(r"/media/skymap/Backup/indo_sinamas_month6/label_v2/W04_202004021432_RI_TPG_TPGD198B01_RGB.shp"
    ,r"/media/skymap/Backup/indo_sinamas_month6/image/W04_202004021432_RI_TPG_TPGD198B01_RGB.tif"
    ,r"/media/skymap/Backup/indo_sinamas_month6/test.shp")

