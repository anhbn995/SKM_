import os
import argparse
from glob import glob
from src.predict_cloud_and_shadow import predict_cloud_shadow
from src.predict_cloud import predict_cloud
from src.predict_shadow import predict_shadow
from src.export_cloud_to_nodata import clip_image_by_mask, clip_image_by_mask_v2
import os
import sys
from glob import glob
import ntpath
import tqdm
import rasterio
import numpy as np
from params import INPUT_PATH, OUTPUT_PATH, ROOT_DATA_FOLDER, TMP_PATH

def combine_mask(first_mask, second_mask, output_mask):
    '''
        Combine segment binary mask from cloud and shadow predictions
    '''
    with rasterio.open(first_mask) as m1:
        with rasterio.open(second_mask) as m2:
            mask1 = m1.read()
            mask2 = m2.read()
            meta = m1.meta
    mask_all = ((mask1+mask2>2)+1).astype(np.uint8)    
    with rasterio.open(output_mask, "w", compress='lzw', **meta) as dest:
        dest.write(mask_all)
    print('Process successfully, output saved at', output_mask)
        

def run_single(im_path, out_path, dir_tmp, cloud_weight, shadow_weight):#cloud_shadow_weight
    os.makedirs(dir_tmp, exist_ok=True)
    for file in glob(im_path):
        fname = os.path.basename(file)
        '''
            mask_tmp1: cloud segment mask
            mask_tmp2: shadow segment mask
            mask_tmp3: union of cloud and shadow segment mask

            out_path1: clip image version 1 - only clipped out cloud
            out_path2: clip image version 2 - clipped out cloud and shadow
        '''
        mask_tmp1 = os.path.join(dir_tmp, fname.replace(".tif","_mask1.tif"))
        mask_tmp2 = os.path.join(dir_tmp, fname.replace(".tif","_mask2.tif"))
        mask_tmp3 = os.path.join(dir_tmp, fname.replace(".tif","_mask3.tif"))

        out_path1 = os.path.join(out_path, fname.replace(".tif","_v1.tif"))
        out_path2 = os.path.join(out_path, fname.replace(".tif","_v2.tif"))
        predict_cloud(img_data=file, output_im = mask_tmp1, weight = cloud_weight)
        clip_image_by_mask(img_path = file, mask_path = mask_tmp1, out_path = out_path1)
        # fixed
        # predict_cloud_shadow(img_data=out_path1, output_im=mask_tmp2, weight = cloud_shadow_weight)
        predict_shadow(img_data = file, output_im = mask_tmp2, weight=shadow_weight)
        combine_mask(mask_tmp1, mask_tmp2, output_mask = mask_tmp3)
        clip_image_by_mask_v2(img_path = file, mask_path=mask_tmp3, out_path=out_path2)
    return out_path1, out_path2

def main(input_txt, output_dir, weight_cloud_only, weight_shadow_only, tmp_path):
    """
        input_txt: File txt chua nhung duong dan file
        output_dir: Thu muc chua ket qua
    """
    with open(input_txt) as f:
        lines = [line.rstrip('\n') for line in f]

    cloud_free_paths = []
    shadow_free_paths = []
    for im_path in lines:
        out_path1, out_path2 = run_single(im_path, output_dir, tmp_path, weight_cloud_only, weight_shadow_only)#weight_cloud_shadow
        cloud_free_paths.append(out_path1)
        shadow_free_paths.append(out_path2)
    out_txt = os.path.join(output_dir, "outdir.txt")

    list_outpath = shadow_free_paths+cloud_free_paths
    with open(out_txt, 'w') as f:
        for line in list_outpath:
            f.write(f"{line}\n")

if __name__ == "__main__":
    input_path = INPUT_PATH
    output_path = OUTPUT_PATH
    tmp_path = TMP_PATH

    weight_cloud_only = f'{ROOT_DATA_FOLDER}/pretrain-models/cloud-free-mosaic-v2/v2/weights/cloud_only.h5'
    weight_shadow_only = f'{ROOT_DATA_FOLDER}/pretrain-models/cloud-free-mosaic-v2/v2/weights/shadow_only.h5'
    # weight_cloud_shadow = f'{ROOT_DATA_FOLDER}/pretrain-models/cloud-free-mosaic-v2/v2/weights/cloud_shadow.h5'
    main(input_path, output_path, weight_cloud_only, weight_shadow_only, tmp_path)
    print("Finished!!!")
    import sys
    sys.exit()