from rio_tiler.io import COGReader
import rasterio
import numpy as np
import gdal
from pyproj import Proj, transform


def get_quantile_schema(img):
    qt_scheme = []
    with COGReader(img) as cog:
        stats = cog.stats()
        for _, value in stats.items():
            qt_scheme.append({
                'p2': value['pc'][0],
                'p98': value['pc'][1],
            })
    return qt_scheme


def remove_nan_value(image_path, new_image):
    src = rasterio.open(image_path)
    out_profile = src.meta
    dst = rasterio.open(new_image, 'w', **out_profile)
    for _, window in src.block_windows(1):
        src_block = src.read(window=window)
        src_block[np.isnan(src_block)] = 0
        dst.write(src_block, window=window)
    src.close()
    dst.close()


def resize_with_resolution_in_meter(in_path, out_path, resolution):
    in_proj = Proj(init='epsg:3857')
    out_proj = Proj(init='epsg:4326')
    x1, y1 = resolution, resolution
    x2, y2 = transform(in_proj, out_proj, x1, y1)

    gdal.Warp(out_path, in_path,
              format='GTiff',
              xRes=x2, yRes=y2,
              resampleAlg=gdal.GRA_Average)
