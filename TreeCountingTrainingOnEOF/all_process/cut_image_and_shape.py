import rasterio
import numpy as np
import glob, os
from itertools import product
import rasterio, warnings
from rasterio import windows
import pandas as pd
from shapely.geometry import Polygon, box, mapping, Point, LineString
from shapely.strtree import STRtree
import geopandas as gp
from tqdm import tqdm
from get_image_resolution_meter import get_resolution_meter
warnings.filterwarnings("ignore")

def get_tiles(ds, width, height, stride):
    ncols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, ncols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)
    offset = []
    for col_off, row_off in offsets:
        if row_off + width > nrows:
            row_off = nrows - width
        if  col_off + height > ncols:
            col_off = ncols - height
        offset.append((col_off, row_off))
    offset = set(offset)
    for col_off, row_off in tqdm(offset): 
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform
        
image_path = '/mnt/Nam/Buildinggg/data_train_v1/image/'
label_path = '/mnt/Nam/Buildinggg/data_train_v1/shape/'
if not os.path.exists(label_path):
    os.makedirs(label_path)
if not os.path.exists(image_path):
    os.makedirs(image_path)
output_box = '{}_{}'

# img_path = '/mnt/Nam/public/tmp_Nam/Building Footprint Data/building_sunny/image/10seg775355.tif'
# shp_path = '/mnt/Nam/public/tmp_Nam/Building Footprint Data/building_sunny/label/10seg775355.shp'

idx = len(os.listdir(image_path))
print(idx)

for img_path in glob.glob("E:\TreeCounting_MasterModel\*.tif"):
    print(img_path)
    shp_path = img_path.replace("image","shape").replace(".tif", ".shp")
    image_resolution = int(round(0.3*512/get_resolution_meter(img_path)/64)*64)
    stride_size = int(image_resolution*3/4)
    name_image = os.path.basename(img_path).replace(".tif","")
    with rasterio.open(img_path) as inds:
        _width, _height = image_resolution, image_resolution
        meta = inds.meta.copy()
        out_crs = inds.crs.to_string()
        my_geoseries = gp.read_file(shp_path)
        my_geoseries["geometry"] = my_geoseries.geometry.buffer(0)
        my_geoseries = my_geoseries.to_crs(out_crs)
        b=STRtree(my_geoseries['geometry'])

        for window, transform in get_tiles(inds, _width, _height, stride_size):
            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height
            min = transform * (0, 0)
            max = transform * (_width, _height)
            boxx = box(*np.minimum(min,max), *np.maximum(min,max))
            # geometry = [o.intersection(boxx) for o in b.query(boxx) if o.intersects(boxx)]
            geometry = []
            for o in b.query(boxx):
                if o.intersects(boxx):
                    intersec = o.intersection(boxx)
                    if intersec.type=="Polygon":
                        if round((intersec.area/transform[0]/-transform[4]),2)>20:
                            geometry.append(intersec)
                    else:
                        try:
                            for i in intersec:
                                if round((i.area/transform[0]/-transform[4]),2)>20:
                                    geometry.append(i)
                        except:
                            print(intersec)

            if 3<len(geometry)<300:
                outpath_image = os.path.join(image_path, output_box.format(name_image,'{0:0004d}'.format(idx)))+ '.tif'
                outpath_label = os.path.join(label_path, output_box.format(name_image,'{0:0004d}'.format(idx)))+ '.shp'
                data_fame = pd.DataFrame(geometry, columns=['geometry'])
                gdf = gp.GeoDataFrame(data_fame, geometry='geometry',crs=out_crs)
                gdf.to_file(outpath_label)
                with rasterio.open(outpath_image, "w", **meta, compress='lzw') as dest:
                    dest.write(inds.read(window=window))
                idx+=1