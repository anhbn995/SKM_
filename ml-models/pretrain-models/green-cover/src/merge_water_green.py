import os
import glob
import rasterio
import numpy as np

def combine_all(path_image, tmp_path, result_green, result_water, result_color):
    # if os.path.exists(result_color):
    #     os.remove(result_color)

    if not os.path.exists(result_color):
        with rasterio.open(path_image) as src:
            msk = src.read_masks(1)
            out_meta = src.meta
        # msk = msk
        mask_green = result_green
        mask_water = result_water
        # print("###",np.unique(mask_water))

        mask_water[mask_water==1]=2
        mask_water[mask_green==1]=1
        mask_water[mask_water==0]=3
        mask_water[msk==1]=0
        mask_all = mask_water

        # print("@@@",np.unique(mask_water))
        combine_path = os.path.join(tmp_path ,os.path.basename(path_image).replace('.tif','_combine.tif'))
        with rasterio.Env():
            profile = out_meta
            profile.update(
                dtype=rasterio.uint8,
                count=1,
                compress='lzw',
                nodata=0)
            with rasterio.open(combine_path, 'w', **profile) as dst:
                dst.write(mask_all.astype(np.uint8),1)

        with rasterio.Env():   
            with rasterio.open(combine_path) as src:  
                shade = src.read()[0]
                meta = src.meta.copy()
                meta.update({'nodata': 0, 'dtype':'uint8'})
            with rasterio.open(result_color, 'w', **meta) as dst:
                dst.write(shade, indexes=1)
                dst.write_colormap(1, {
                        0: (0,0,0, 0),
                        1: (34,139,34,0), #Green1
                        # 2: (157,193,131,0), #Green2
                        2: (100, 149, 237, 0), #water
                        3: (101,67,33, 0)}) #Buildup
        return result_color

if __name__ == "__main__":
    # name = 'T11'
    path = '/home/quyet/data/bk_2/results'
    list_folder = sorted(os.listdir(path))
    # list_month = glob.glob(os.path.join(path, '20*'))
    # print(list_month)
    for j in list_folder:
        aaa = os.path.join(path, j)
        bbb = sorted(os.listdir(aaa))
        bbb = [bbb[0]]
        print(bbb)
        for k in bbb:
            folder_path =os.path.join(aaa, k)
            list_img = sorted(glob.glob(os.path.join(folder_path, "*.tif")))
            path_image = list_img[0]
            path_green = path_image.replace('.tif', '_green.tif')
            path_water = path_image.replace('.tif', '_water.tif')
            result_green = rasterio.open(path_green).read()[0]
            result_water = rasterio.open(path_water).read()[0]
            print(folder_path)
            combine_all(path_image, result_green, result_water)

    # for i in list_folder:
    #     folder_path = os.path.join(path, i)
    #     print(folder_path)
    #     list_img = sorted(glob.glob(os.path.join(folder_path, "*.tif")))
    #     path_image = list_img[0]
    #     path_green = list_img[2]
    #     path_water = list_img[3]
    #     result_green = rasterio.open(path_green).read()[0]
    #     result_water = rasterio.open(path_water).read()[0]
    #     combine_all(path_image, result_green, result_water)