import fiona
import rasterio 
import numpy as np
import rasterio.mask
import matplotlib.pyplot as plt

from pyproj import CRS
from fiona.crs import from_string
from shapely.geometry import Polygon, mapping


def write_image(data, height, width, numband, crs, tr, out):
    """
        Export numpy array to image by rasterio.
    """
    with rasterio.open(out,'w',driver='GTiff',height=height,
                        width=width,count=numband,dtype=data.dtype,crs=crs,
                        transform=tr,) as dst:
                        dst.write(data)

def filter_cloud(msk, path_mask, height, width, src_transform):
    msk_1 = msk + 1

#     with fiona.open(path_mask, "r") as shapefile:
#         features = [f["geometry"] for f in shapefile]
        
#     mask_shp = rasterio.features.geometry_mask(features, (height, width), src_transform, 
#                                                 invert=True, all_touched=True).astype(np.uint8)
    with rasterio.open(path_mask) as src:
        mask_shp = src.read()
    mask_test = mask_shp.astype(np.float32) - msk_1.astype(np.float32)
    # mask_test = np.ones_like(mask_test) - mask_test
    # mask_test = mask_test.astype(np.uint8)

    mask_test = ((mask_test + 1)/2).astype(np.int16)
    # mask_test = ((mask_test  -1)*-1 + 0)
    mask_test = mask_test.astype(np.uint8)
    return mask_test

def combine_one(path_image, path_mask_1, path_mask_2, name_output, convert_tiff=False):
    with rasterio.open(path_image) as src:
        height = src.height
        width = src.width
        crs =src.crs
        src_transform = src.transform
        msk = src.read_masks(1)

    mask_water = filter_cloud(msk, path_mask_1, height, width, src_transform)
    mask_water = (mask_water*2).astype(np.uint8)

#     with fiona.open(path_mask_2, "r") as shapefile:
#         features = [f["geometry"] for f in shapefile]
        
#     mask_green = rasterio.features.geometry_mask(features, (height, width), src_transform, 
#                                                  invert=True, all_touched=True).astype(np.uint8)
    
    with rasterio.open(path_mask_2) as src:
        mask_green = src.read()
        
    mask_green = np.ones_like(mask_green) - mask_green
    mask_green = (mask_green*3).astype(np.uint8)

    msk_cloud = msk/255
    msk_cloud = (msk_cloud -1)*-1 +0 
    msk_cloud = msk_cloud.astype(np.uint8)

    img = msk_cloud  + mask_water + mask_green
    img[img==4] = 1
    img[img==5] = 2
    img[img==0] = 6
    img[img==1] = 0
    img[img==6] = 1
    
    if convert_tiff:
        write_image(img[0].astype(np.uint8)[np.newaxis,...], height, width, 1, 
                    crs, src_transform, name_output)
    else:
        shapes_1 = rasterio.features.shapes(mask_water, transform=src_transform)
        shapes_2 = rasterio.features.shapes(mask_green, transform=src_transform)
        shapes_3 = rasterio.features.shapes(msk_cloud, transform=src_transform)
        img1_1 = img
        img1_1[img1_1==3] = 4
        img1_1[img1_1!=4] = 0
        msk_non = img1_1
        shapes_4 = rasterio.features.shapes(msk_non, transform=src_transform)
        shapes = []
        for s in shapes_1:
            if s[-1]>0:
                shapes.append([s[0]['coordinates'], 2])
        for s in shapes_2:
            if s[-1]==0:
                shapes.append([s[0]['coordinates'], 3])
        for s in shapes_3:
            if s[-1]>0:
                shapes.append([s[0]['coordinates'], 0])
        for s in shapes_4:
            if s[-1]>0:
                shapes.append([s[0]['coordinates'], 1])
        
        class_name = ['cloud', 'non-green', 'water', 'green']

        schema = {
                'geometry': 'Polygon',
                'properties': {'label': 'str'},}
        name_shp = name_output.replace('.tif', '.shp')
        crs = CRS.from_epsg(crs.to_epsg()).to_proj4()

        with fiona.open(name_shp, 'w', driver='ESRI Shapefile', schema=schema, crs=from_string(crs)) as c:
            for poly in shapes:
                box, label = poly
        #         print(label)
                c.write({
                    'geometry': mapping(Polygon(box[0])),
                    'properties': {'label': class_name[int(label)]}
                })

    return img, mask_water, mask_green, msk_cloud

