import os, glob
import rasterio
import numpy as np
import geopandas as gpd
import rasterio.features
from distutils.dir_util import copy_tree

def standardized_shape_epsg(fp_shp, fp_img, out_dir):
    output_path = os.path.join(out_dir, os.path.basename(fp_shp).replace('.shp', '_std.shp'))
    os.makedirs(out_dir, exist_ok=True)
    df_shape = gpd.read_file(fp_shp)
    
    with rasterio.open(fp_img, mode='r+') as src:
        projstr = src.crs.to_string()
        check_epsg = src.crs.is_epsg_code
        if check_epsg:
            epsg_code = src.crs.to_epsg()
        else:
            epsg_code = None
        if epsg_code:
            out_crs = {'init':'epsg:{}'.format(epsg_code)}
        else:
            out_crs = projstr
    gdf = df_shape.to_crs(out_crs)
    gdf.to_file(output_path)
        

def clip_aoi_image(fp_img, fp_aoi, out_dir_img_cut):
    os.makedirs(out_dir_img_cut, exist_ok=True)
    img_name = os.path.basename(fp_img).replace('.tif','')
    with rasterio.open(fp_img, mode='r') as src:
        projstr = src.crs.to_string()
        print(projstr)
        check_epsg = src.crs.is_epsg_code
        if check_epsg:
            epsg_code = src.crs.to_epsg()
            print(epsg_code)
        else:
            epsg_code = None
    if epsg_code:
        out_crs = {'init':'epsg:{}'.format(epsg_code)}
    else:
        out_crs = projstr
        
    bound_shp = gpd.read_file(fp_aoi)
    bound_shp = bound_shp.to_crs(out_crs)

    for index2, row_bound in bound_shp.iterrows():
        geoms = row_bound.geometry
        img_cut = img_name+"_{}.tif".format(index2)
        img_cut_path = os.path.join(out_dir_img_cut, img_cut)
        try:
            with rasterio.open(fp_img, BIGTIFF='YES') as src:
                out_image, out_transform = rasterio.mask.mask(src, [geoms], crop=True)
                out_meta = src.meta
            out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})
            with rasterio.open(img_cut_path, "w", **out_meta) as dest:
                dest.write(out_image)
        except Exception:
            print(img_name, index2)
    return True    
    

def create_mask_by_shape(fp_shp, fp_img, fp_mask_out):
    df_shp = gpd.read_file(fp_shp)
    list_geo = [(x.geometry) for i, x in df_shp.iterrows()]
    
    
    with rasterio.open(fp_img) as src:
        tr = src.transform
        width, height = src.width, src.height
        meta = src.meta
    
    mask = rasterio.features.geometry_mask(list_geo,
                                           out_shape=(height, width),
                                           transform=tr,
                                           invert=True,
                                           all_touched=True).astype('uint8')
    meta.update({
        'dtype': 'uint8',
        'count': 1
    })
    with rasterio.open(fp_mask_out, 'w', **meta) as dst:
        dst.write(np.array([mask]))


def move_folder_remove_json_file(src_folder, dst_folder):
    copy_tree(src_folder, dst_folder)
    # tìm file json và xóa
    list_json = glob.glob(os.path.join(dst_folder, '*/*.json'))
    for fp_json in list_json:
        os.remove(fp_json)
    print(f'Copy done {os.path.basename(src_folder)}')


def move_list_dir_to_dst_dir(list_src_dir, dst_dir):
    for src_dir in list_src_dir:
        move_folder_remove_json_file(src_dir, dst_dir)


if __name__=='__main__':
    image_dir = r'E:\TMP_XOA\mongkos\mongkos part 3_transparent_mosaic_group1.tif'
    label_dir = r'E:\TMP_XOA\mongkos_2\label.shp'
    box_dir = r'E:\TMP_XOA\mongkos_2\box.shp'
    out_dir = r'E:\TMP_XOA\mongkos_std'
    standardized_shape_epsg(label_dir, image_dir, out_dir)