import os, glob
import rasterio
import numpy as np
from tqdm import tqdm


def remove_img_nodata(fp_img_mask_check, fp_img):
    with rasterio.open(fp_img_mask_check) as src:
        img = src.read()
        number_pixel = src.height*src.width
    if np.all(img == 0):
        if os.path.exists(fp_img_mask_check) and os.path.exists(fp_img):
            os.remove(fp_img_mask_check)
            os.remove(fp_img)
        else:
            print(f"The file img or mask does not exist with name {os.path.basename(fp_img)}")


def get_list_name_fp(folder_dir, type_file = '*.tif'):
        """
            Get all file path with file type is type_file.
        """
        list_fp = []
        for file_ in glob.glob(os.path.join(folder_dir, type_file)):
            head, tail = os.path.split(file_)
            # list_fp.append(os.path.join(head, tail))
            list_fp.append(tail)
        return list_fp

if __name__ == "__main__":
    """Chay 1 cai"""
    fd_img = r"/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/data_faster-rnn/image_crop_box/tmp/gen_TauBien_Original_3band_cut_128_stride_64_time_20230710_011308/images"
    fd_img_mask_check = r"/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/data_faster-rnn/image_crop_box/tmp/gen_TauBien_Original_3band_cut_128_stride_64_time_20230710_011308/masks"
        
    list_name = get_list_name_fp(fd_img_mask_check)

    for name in tqdm(list_name):
        fp_img = os.path.join(fd_img, name)
        fp_mask = os.path.join(fd_img_mask_check, name)
        remove_img_nodata(fp_mask, fp_img)


