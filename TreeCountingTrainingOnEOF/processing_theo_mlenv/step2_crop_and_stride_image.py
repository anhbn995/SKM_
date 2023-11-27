import os, glob
import rasterio
import numpy as np
from tqdm import tqdm
from rasterio.windows import Window
# from tqdm.notebook import tqdm_notebook
from remove_nodata_and_hold_percent import remove_file_nodata_on_img_and_mask

def crop_img_stride(image_path, outdir_crop, crop_size, stride_size):
    name_base = os.path.basename(image_path)
    i = 0
    with rasterio.open(image_path) as src:
        h,w = src.height,src.width
        meta = src.meta
        list_weight = list(range(0, w, stride_size))
        list_hight = list(range(0, h, stride_size))

        with tqdm(total=len(list_hight)*len(list_weight)) as pbar:
            for start_h_org in list_hight:
                for start_w_org in list_weight:
                    win = Window(start_w_org, start_h_org, crop_size, crop_size)
                    img_window_crop  = src.read(window=win)
                    win_transform = src.window_transform(win)
                    meta.update({'height': crop_size, 'width': crop_size, 'transform':win_transform, 'nodata': 0})
                    name_file = name_base.replace(".tif", f"_{i}.tif")
                    fp_out = os.path.join(outdir_crop, name_file)
                    with rasterio.open(fp_out, 'w',**meta) as dst:
                        dst.write(img_window_crop, window=Window(0, 0, img_window_crop.shape[2], img_window_crop.shape[1]))
                    i+=1
                    pbar.update(1)

if __name__ == "__main__":


    """Chay 1folder img khong chay mask"""
    # dir_img = r'/home/skm/SKM16/3D/Data/Gen_512_stride56/height_1024'
    # outdir_crop = r'/home/skm/SKM16/3D/Data/Gen_512_stride56/height_512'
    # crop_size = 512
    # stride_size = 100
    # for image_path in tqdm(glob.glob(os.path.join(dir_img, "*.tif"))):
    #     # image_path = r"E:\WorkSpaceSkyMap\Change_detection_Dubai\Data\tach_ra\v2\B\data_change.tif"
    #     # outdir_crop = f"E:\WorkSpaceSkyMap\Change_detection_Dubai\Data\/tach_ra\/v2\cut{crop_size}stride{stride_size}\B"
    #     # print(outdir_crop)
    #     os.makedirs(outdir_crop, exist_ok=True)
    #     crop_img_stride(image_path, outdir_crop, crop_size, stride_size)
    
    """Chay 1 img 1 mask"""
    from crop_and_stride_image_config import *
    for folder_name in name_folder_img_and_mask:
        print("Processing folder: ", folder_name)
        dir_img = os.path.join(dir_img_and_mask, folder_name)
        list_fname = [os.path.basename(fp) for fp in glob.glob(os.path.join(dir_img, '*.tif'))]
        outdir_crop = os.path.join(dir_img_and_mask, f"gen_{name_project}",f"{rename_folder[name_folder_img_and_mask.index(folder_name)]}")
        os.makedirs(outdir_crop, exist_ok=True)
        for fname in list_fname:
            print(f'Processing {fname}')
            image_path = os.path.join(dir_img, fname)
            crop_img_stride(image_path, outdir_crop, crop_size, stride_size)
    # print(f"\n ===> Xoa mask toan khong giu lai {phan_tram_giu_lai_mask_0}% mask toan den")
    
    # remove_file_nodata_on_img_and_mask(dir_img, dir_mask_label, percent_nodata_trong_datatrain, percent_quyet_dinh_la_nodata)
    
    """Chay nhieu cai cho change detection"""
    # for folder_name in ['A', 'B', 'label']:
    #     dir_img = f"/home/skm/SKM16/Data/WORK/MSRAC/Unstak/uint8/Img_cut/{folder_name}"
    #     list_fname = [os.path.basename(fp) for fp in glob.glob(os.path.join(dir_img, '*.tif'))]
    #     outdir_crop = f"/home/skm/SKM16/Data/WORK/MSRAC/Unstak/uint8/Img_cut/zgen_cut{crop_size}stride{stride_size}/{folder_name}"
    #     os.makedirs(outdir_crop, exist_ok=True)
    #     for fname in list_fname:
    #         print(f'Processing {fname}')
    #         image_path = os.path.join(dir_img, fname)
    #         crop_img_stride(image_path, outdir_crop, crop_size, stride_size)