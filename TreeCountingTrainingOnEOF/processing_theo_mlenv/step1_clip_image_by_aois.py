import os
import rasterio
from tqdm import tqdm
import geopandas as gpd
from rasterio.mask import mask


fp_need_cut = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/img_ori/2023-06_mosaic_cog.tif'
fp_shp_cut = r'/home/skm/SKM16/Planet_GreenChange/a_4326.shp' #a_4326.shp' , aaaaaa.geojson
out_dir = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/img_ori/clip'
os.makedirs(out_dir, exist_ok=True)


aois = gpd.read_file(fp_shp_cut)
with rasterio.open(fp_need_cut) as src:
    for index in range(len(aois)):
        print(index)
        list_polygon = [aois.geometry[index]]
        out_image, out_transform = mask(src, list_polygon, crop=True)
        print(index)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})
        out_fp = os.path.join(out_dir, os.path.basename(fp_need_cut).replace('.tif', f'_{index}.tif'))
        with rasterio.open(out_fp, "w", **out_meta) as dest:
            dest.write(out_image)
        print("oke")

