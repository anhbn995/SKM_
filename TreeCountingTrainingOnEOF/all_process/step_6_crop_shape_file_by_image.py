import numpy as np
from pyproj import transform,Proj, Transformer
import cv2
import glob, os
import sys
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
import time
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from shapely.geometry.multipolygon import MultiPolygon
import geopandas as gp
import pandas as pd
# from shapely.ops import transform
from tqdm import *
import rasterio

core = multiprocessing.cpu_count()*3//4
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

def load_shapefile(shape_path):
    data_shp = gp.read_file(shape_path)
    gpdf_singlepoly = multi2single(data_shp)
    return gpdf_singlepoly

def polygon_to_geopolygon(polygon, geotransform):
    topleftX = geotransform[2]
    topleftY = geotransform[5]
    XRes = geotransform[0]
    YRes = geotransform[4]
    poly = np.array(polygon)
    poly_rs = poly*np.array([XRes,YRes])+np.array([topleftX,topleftY])
    return poly_rs

def load_image_geom(image_path):
    with rasterio.open(image_path) as src:
        geotransform1 = src.transform
        w,h = src.width,src.height
        projstr = (src.crs.to_string())
        epsg_code = src.crs.to_epsg()
    polygon = ((0,0),(w,0),(w,h),(0,h),(0,0))
    geopolygon = polygon_to_geopolygon(polygon,geotransform1)
    list_point = []
    for point in geopolygon:
        long1,lat = point[0],point[1]
        x,y = long1,lat
        list_point.append((x,y))
    return Polygon(tuple(list_point))

def create_list_id(path,shape_name):
    list_id = []
    os.chdir(path)
    len_id = len(shape_name)
    for file in glob.glob("*.tif"):
        # list_id.append(file[:-4])
        if file[0:len_id]==shape_name:
            list_id.append(file[:-4])
        # print(file[:-4])
    return list_id

def create_list_id_shape(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.shp"):
        list_id.append(file[:-4])
        # if file[0:6]=='SX9192':
            # list_id.append(file[:-4])
        # print(file[:-4])
    return list_id

def crop_shape_pool(id_image,gdf_building,path_shape_crop,foder_image):
    result2 = load_image_geom(os.path.join(foder_image,id_image+'.tif'))
#     gdf_building_cliped = gdf_building.clip(result2)
    crs = gdf_building.crs
    result2_id = list(range(len([result2])))
    data_box = list(zip([result2], result2_id))
    data_fame = pd.DataFrame(data_box, columns=['geometry','FID'])
    gdf_box = gp.GeoDataFrame(data_fame, geometry='geometry',crs=crs)
    gdf_building_cliped = gp.overlay(gdf_building, gdf_box, how='intersection')
    gdf_building_cliped = multi2single(gdf_building_cliped)
#     crs2 = {'init':'epsg:3857'}
#     gdf_building_cliped = gdf_building_cliped.to_crs(crs2)
#     gdf_building_cliped['area'] = gdf_building_cliped.area
#     gdf_building_cliped = gdf_building_cliped[gdf_building_cliped['area'] > 10]
    shp_out_path = (os.path.join(path_shape_crop,id_image+'.shp'))
    try:
#         gdf_building_cliped = gdf_building_cliped.to_crs(crs)
        gdf_building_cliped.to_file(shp_out_path)
    except:
        pass
    return True
def crop_shape2(foder_image,shape_name,shape_dir):
    shape_path=os.path.join(shape_dir,shape_name+'.shp')
    gdf_building = load_shapefile(shape_path)
    list_id = create_list_id(foder_image,shape_name)
    foder_name = os.path.basename(foder_image)
    parent = os.path.dirname(foder_image)
    if not os.path.exists(os.path.join(parent,foder_name+'_shape')):
        os.makedirs(os.path.join(parent,foder_name+'_shape'))
    path_shape_crop = os.path.join(parent,foder_name+'_shape')
#     for i in tqdm(range(len(list_id))):
#         id_image = list_id[i]
#         crop_shape_pool(id_image,gdf_building,path_shape_crop,foder_image)
    p_cropshape = Pool(processes=core)
    pool_result = p_cropshape.imap_unordered(partial(crop_shape_pool,gdf_building=gdf_building,path_shape_crop=path_shape_crop,foder_image=foder_image), list_id)
#     with tqdm(total=len(list_id)) as pbar:
    for i,_ in tqdm(enumerate(pool_result)):
        pass
#             pbar.update()
    p_cropshape.close()
    p_cropshape.join()
def main_crop_shape(foder_image,shape_dir):
    x1 = time.time()
    list_shape = create_list_id_shape(shape_dir)
    for shape_name in list_shape:
        crop_shape2(foder_image,shape_name,shape_dir)
    print(time.time() - x1, "second")
    foder_name = os.path.basename(foder_image)
#  size_crop = int(sys.argv[3])
    parent = os.path.dirname(foder_image)
    path_shape_crop = os.path.join(parent,foder_name+'_shape')
    return path_shape_crop
if __name__ == "__main__":
    x1 = time.time()
    foder_image = r'/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/data_faster-rnn/image_crop_box/tmp/gen_TauBien_Original_3band_cut_256_stride_128_V2/images_data/train/images'
    shape_dir = r"/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/ShipDetection/label2" #os.path.abspath(sys.argv[2])
    list_shape = create_list_id_shape(shape_dir)
    for shape_name in list_shape:
        crop_shape2(foder_image,shape_name,shape_dir)
    print(time.time() - x1, "second")
