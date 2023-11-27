# -*- coding: utf-8 -*-
"""
Created on Wed May 24 16:32:39 2023

@author: DucAnh
"""

import os
import rasterio
import numpy as np
from rasterio.warp import transform_bounds


def get_bbox_intersect_2_img(list_fp_img):
    list_bound = []
    list_crs = []
    for fp_img in list_fp_img:
        with rasterio.open(fp_img) as src:
            bounds = src.bounds
            crs = src.crs
        list_bound.append(bounds)
        list_crs.append(crs)
    
    bound_left = [bounds.left for bounds in list_bound]
    bound_bottom = [bounds.bottom for bounds in list_bound]
    bound_right = [bounds.right for bounds in list_bound]
    bound_top = [bounds.top for bounds in list_bound]

    xmin = max(bound_left)
    ymin = max(bound_bottom)
    xmax = min(bound_right) 
    ymax = min(bound_top)

    leftw, bottomw, rightw, topw = transform_bounds(list_crs[0], list_crs[1], xmin, ymin, xmax, ymax)
    left, bottom, right, top = transform_bounds(list_crs[1], list_crs[0], leftw, bottomw, rightw, topw)
    return (left, bottom, right, top)


def clip_raster_by_bbox(input_path, bbox, output_path= None, return_ = True, export_file_tiff = True):
    with rasterio.open(input_path) as src:
        minx, miny, maxx, maxy = bbox
        window = src.window(minx, miny, maxx, maxy)
        
        width = window.width
        height = window.height
        
        transform = rasterio.windows.transform(window, src.transform)
        nodata = src.nodata
        
        meta = src.meta.copy()
        meta.update({
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'transform': transform
        })
        if export_file_tiff:
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(src.read(window=window))
        if return_:
            return src.read(window=window), meta, nodata
 
def main(fp_in_truoc, fp_in_sau, fp_out):
    os.makedirs(os.path.dirname(fp_out), exist_ok=True)
    bbox = get_bbox_intersect_2_img([fp_in_truoc, fp_in_sau])
    img_truoc, meta, nodata1 = clip_raster_by_bbox(fp_in_truoc, bbox, return_ = True, export_file_tiff = False)
    img_sau, meta, nodata2 = clip_raster_by_bbox(fp_in_sau, bbox, return_ = True, export_file_tiff = False)
    meta.update({'count':1})
    ind_nodata_truoc = np.where(img_truoc==nodata1)
    ind_nodata_sau = np.where(img_sau==nodata2)
    change = np.empty_like(img_sau)
    change = img_sau - img_truoc
    change[ind_nodata_truoc] = nodata1
    change[ind_nodata_sau] = nodata2
    
    with rasterio.open(fp_out, 'w', **meta) as dst:
        dst.write(change)
        
if __name__=='__main__':
    fp_in_truoc = r'/home/skm/SKM16/Tmp/XONG_XOAAAAAAAAAAAAAAAAAAAAAAAA/Img/S1A_IW_GRDH_1SDV_20220323T215024_0.tif'
    fp_in_sau = r'/home/skm/SKM16/Tmp/XONG_XOAAAAAAAAAAAAAAAAAAAAAAAA/Img/S1A_IW_GRDH_1SDV_20220416T215025_0.tif'
    fp_out = r'/home/skm/SKM16/Tmp/XONG_XOAAAAAAAAAAAAAAAAAAAAAAAA/Img/rs/change20220323_vs_20220416_v2100.tif'
    main(fp_in_truoc, fp_in_sau, fp_out)
    print('DONE')
