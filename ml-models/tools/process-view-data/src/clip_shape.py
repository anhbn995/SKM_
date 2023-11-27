# import rasterio
from osgeo import gdal
import glob, os
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

from tqdm import *
import multiprocessing
from functools import partial
from multiprocessing.pool import Pool

core = multiprocessing.cpu_count()*3//4

def get_fn_file(dir_img):
    return [os.path.basename(fp).replace('.tif','')  for fp in glob.glob(os.path.join(dir_img,'*.tif'))]


def load_image_geom(fp_img):
    ds = gdal.Open(fp_img)
    x_min, x_size, _, y_max, _, y_size = ds.GetGeoTransform()
    x_max = x_min + x_size * ds.RasterXSize
    y_min = y_max + y_size * ds.RasterYSize
    poly = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])
    return poly


def clip_shape_by_image(image_path, df_shape_label, fp_shp_out):
    polygon_clip = load_image_geom(image_path)
    gdf_clip = gpd.clip(df_shape_label, polygon_clip)
    gdf_clip.to_file(fp_shp_out)

    


from task import task_proccessing_percent
import time

def clip_all_shape(dir_img, fp_shp, dir_out_shp):
    gdf_lable = gpd.read_file(fp_shp)
    os.makedirs(dir_out_shp, exist_ok=True)
    list_fname = get_fn_file(dir_img)

    def clip_shape_by_folder_image_pool(fn_image, dir_img_path, df_shape_label, dir_shp_out):
        fp_img = os.path.join(dir_img_path, fn_image + '.tif')
        fp_shp_out = os.path.join(dir_shp_out, fn_image + '.shp')
        clip_shape_by_image(fp_img, df_shape_label, fp_shp_out)


    # with ThreadPoolExecutor(max_workers=16) as worker:
    percentage = 0.3
    percentage_by_img = round(0.5/len(list_fname),2)
    print(percentage_by_img)
    for path in list_fname:
        clip_shape_by_folder_image_pool(path,dir_img, gdf_lable, dir_out_shp)
        try:
            percentage += percentage_by_img
            task_proccessing_percent(percentage)
            print(percentage)
        except Exception as e:
            pass
    
    
if __name__ == "__main__":    
    dir_img_path = r'E:\TMP_XOA\mongkos_std\img_cut'
    fp_shp_out = r'E:\TMP_XOA\mongkos_std\label\mongkos part 3_transparent_mosaic_group1.shp'
    dir_shp_out = r"E:\TMP_XOA\mongkos_std\mmm"

    import time
    x = time.time()
    clip_all_shape(dir_img_path, fp_shp_out, dir_shp_out)
    print(time.time() - x)