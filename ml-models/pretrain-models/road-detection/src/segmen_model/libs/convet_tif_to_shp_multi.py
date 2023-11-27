import copy
import rasterio
import numpy as np


def split_tiff(mask_path):
    with rasterio.open(mask_path) as src:
        img = src.read().swapaxes(0,1).swapaxes(1,2)
        crs = src.crs
        src_transform = src.transform
        height = src.height
        width = src.width
        out_meta = src.meta
        
    num_class = np.unique(img)
    
    list_img = []
    for i in num_class[:-1]:
        img_1=copy.deepcopy(img[:,:,0])
        img_1[img_1!=(i+1)]=0
        img_1 = img_1.astype(np.uint8)
        np.unique(img_1)
        list_img.append(img_1)
#     print(len(list_img))
    
    for j,k in enumerate(list_img):
        with rasterio.Env():

            # Write an array as a raster band to a new 8-bit file. For
            # the new file's profile, we start with the profile of the source
            profile = out_meta

            # And then change the band count to 1, set the
            # dtype to uint8, and specify LZW compression.
            profile.update(
                dtype=rasterio.uint8,
                count=1,
                compress='lzw')

            with rasterio.open(mask_path.replace('.tif', '%s.tif'%(j+1)), 'w', **profile) as dst:
                dst.write(k.astype(np.uint8),1)