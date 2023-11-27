import os
import numpy as np
import geopandas as gpd
import tensorflow as tf
import rasterio
import rasterio.features

def write_image(data, height, width, numband, crs, tr, out):
    """
        Export numpy array to image by rasterio.
    """
    with rasterio.open(out,'w',driver='GTiff',height=height,
                        width=width,count=numband,dtype=data.dtype,crs=crs,
                        transform=tr,) as dst:
                        dst.write(data)

def build_mask(img_path, shape_path, out_dir):
    with rasterio.open(img_path) as src:
        src_transform = src.transform
        height = src.height
        width = src.width
        crs = src.crs

    df = gpd.read_file(shape_path)
    out_arr = np.zeros((height,width))
    shapes = ((geom, value) for geom, value in zip(df.geometry, df['id'].astype('uint32')))
    band = rasterio.features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=src_transform)
    write_image(data=band[np.newaxis,...], height=height, width=width, 
                    numband=1, crs=crs, tr=src_transform, out=out_dir)
    if os.path.exists(out_dir):
        ROOT_DIR = os.path.dirname(os.path.abspath(out_dir)) 
        out_dir = os.path.join(ROOT_DIR, out_dir.split('/')[-1])
        return out_dir
    else:
        assert os.path.exists(out_dir), "Can't build mask, please check it again."
