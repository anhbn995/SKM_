import os
import json
import glob
import fiona
import rasterio
import numpy as np
import geopandas as gpd

from osgeo import gdal
from shapely.ops import cascaded_union
from shapely.geometry import mapping, Polygon
from rasterio.warp import reproject, Resampling

def get_list_fp(folder_dir, type_file = '*.tif'):
        list_fp = []
        for file_ in glob.glob(os.path.join(folder_dir, type_file)):
            head, tail = os.path.split(file_)
            list_fp.append(os.path.join(head, tail))
        return list_fp

def reproject_image(src_path, dst_path, dst_crs='EPSG:4326'):
    with rasterio.open(src_path) as ds:
        nodata = ds.nodata or 0
    temp_path = dst_path.replace('.tif', 'temp.tif')
    option = gdal.TranslateOptions(gdal.ParseCommandLine("-co \"TFW=YES\""))
    gdal.Translate(temp_path, src_path, options=option)
    option = gdal.WarpOptions(gdal.ParseCommandLine("-t_srs {} -dstnodata {}".format(dst_crs, nodata)))
    gdal.Warp(dst_path, temp_path, options=option)
    os.remove(temp_path)
    return True

def reproject_profile(src_path, dst_path):
    img1 = rasterio.open(src_path)
    meta = img1.meta.copy()

    img2 = rasterio.open(dst_path)
    img_temp = dst_path.replace('.tif', '_new.tif')
    with rasterio.open(img_temp, 'w', **meta) as dst:
        dst.write(img2.read())
    return img_temp
    

def renew_baseimg(base_img_path, crs):
    base_img_path_recrs = base_img_path.replace(os.path.basename(base_img_path),'z_base_recrs.tif')
    reproject_image(base_img_path, base_img_path_recrs, crs)
    return base_img_path_recrs

def write_shp(box_aoi, box_path, name, crs):
    schema = {
            'geometry': 'Polygon',
            'properties': {'id': 'int'},
        }
    name_shp = name+'.shp'
    box_aoi_path = os.path.join(box_path, name_shp)
    aaa = []
    for i in box_aoi:
        # print(i)
        aaa.append(i['geometry']['coordinates'])
    
    polys = []
    for number, j in enumerate(aaa):
        polys.append(Polygon(j[0]))

    boundary = gpd.GeoSeries(cascaded_union(polys))
    with fiona.open(box_aoi_path, 'w', 'ESRI Shapefile', schema, crs='EPSG:4326') as c:
    # with fiona.open(box_aoi_path, 'w', 'ESRI Shapefile', schema, crs=crs) as c:
        c.write({
            'geometry': mapping(boundary[0]),
            'properties': {'id': 0},
        })
    return box_path

def get_crs(folder_paths, list_month):
    namee = os.path.basename(list_month[0])
    crs = rasterio.open(glob.glob(os.path.join(folder_paths, namee,'*.tif'))[0]).crs.to_string()
    return crs

def write_json_file(data, name):
    cur_path = os.getcwd()
    json_file = os.path.join(cur_path, '%s.json'%(name))
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return json_file

def sorted_month_folder(folder_paths):
    list_month = []
    for i in glob.glob(os.path.join(folder_paths,'T*')):
        a = int(os.path.basename(i).replace('T', ''))
        list_month.append(os.path.join(folder_paths, 'T%s'%str(a)))
    
    return list_month

def standard_coord(img_path, crs):
    list_img = glob.glob(os.path.join(img_path, '*.tif'))
    for i in list_img:
        with rasterio.open(i) as dst:
            crs_temp = dst.crs.to_string()
        if crs_temp == crs:
            pass
        else:
            reproject_image(i, i, dst_crs=crs)

def standard_coord(img_path, crs):
    list_img = glob.glob(os.path.join(img_path, '*.tif'))
    for i in list_img:
        with rasterio.open(i) as dst:
            crs_temp = dst.crs.to_string()
        if crs_temp == crs:
            pass
        else:
            reproject_image(i, i, dst_crs=crs)

def window_from_extent(xmin, xmax, ymin, ymax, aff):
        col_start, row_start = ~aff * (xmin, ymax)
        col_stop, row_stop = ~aff * (xmax, ymin)
        return ((int(row_start), int(row_stop)), (int(col_start), int(col_stop)))

def convert_profile(src_path, dst_path, out_path):
    kwargs = None
    _info = gdal.Info(dst_path, format='json')
    xmin, ymin = _info['cornerCoordinates']['lowerLeft']
    xmax, ymax = _info['cornerCoordinates']['upperRight']

    with rasterio.open(dst_path) as dst:
        dst_transform = dst.transform
        kwargs = dst.meta
        kwargs['transform'] = dst_transform
        dst_crs = dst.crs

    with rasterio.open(src_path) as src:
        window = window_from_extent(xmin, xmax, ymin, ymax, src.transform)
        src_transform = src.window_transform(window)
        data = src.read()

        with rasterio.open(out_path, 'w', **kwargs) as dst:
            for i, band in enumerate(data, 1):
                _band = src.read(i, window=window)
                dest = np.zeros_like(_band)
                reproject(
                    _band,
                    dest,
                    src_transform=src_transform,
                    src_crs=src.crs,
                    dst_transform=src_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

                dst.write(dest, indexes=i)

if __name__ == '__main__':
    from cloud_mask.raster_to_vector import raster_to_vector
    workspace = '/home/quyet/data/Jakarta/results'
    list_folder = os.listdir(workspace)
    for i in list_folder:
        folder_path = os.path.join(workspace, i)
        tif_file = sorted(glob.glob(os.path.join(folder_path, '*.tif')))[0].replace('.tif', '_cloud.tif')
        shape_folder = os.path.join(workspace, 'shp')
        if not os.path.exists(shape_folder):
            os.mkdir(shape_folder)
        shp_file = os.path.join(shape_folder, os.path.basename(tif_file).replace('.tif', '.shp'))
        raster_to_vector(tif_file, shp_file)