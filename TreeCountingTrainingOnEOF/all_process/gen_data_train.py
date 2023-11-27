import glob, os
import sys
import argparse
from step_1_data_standardization import main_std
from step_2_cut_image_by_shape import main_cut_img
from step_3_buil_mask import main_build_mask
from step_4_gen_data_with_size import main_gen_data_with_size
from step_5_split_train_val import main_split
from step_6_crop_shape_file_by_image import main_crop_shape
from step_7_gen_anotation import main_gen_anotation
import time
import shutil
def main(image_dir,box_dir,label_dir,out_dir):
    split = 0.6  
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    box_out_dir, label_out_dir = main_std(image_dir,box_dir,label_dir,out_dir)
    img_cut_dir = main_cut_img(image_dir,box_out_dir,out_dir)
    img_cut_mask_dir = main_build_mask(img_cut_dir,label_out_dir)

    out_data_image_crop = os.path.join(out_dir,"image_result_crop")
    out_data_image_crop_base,out_data_image_crop_mask = main_gen_data_with_size(img_cut_dir,img_cut_mask_dir,out_data_image_crop,sampleSize=256)
    out_dir_data_final = os.path.join(out_dir,"data_result")

    path_train,path_val,path_test = main_split(out_data_image_crop_base,out_data_image_crop_mask,split,out_dir_data_final)

    crop_shape_train = main_crop_shape(path_train,label_out_dir)
    crop_shape_val = main_crop_shape(path_val,label_out_dir)
    crop_shape_test = main_crop_shape(path_test,label_out_dir)

    main_gen_anotation(path_train,crop_shape_train)
    main_gen_anotation(path_val,crop_shape_val)
    main_gen_anotation(path_test,crop_shape_test)
    
    shutil.rmtree(box_out_dir)
    shutil.rmtree(label_out_dir)
    shutil.rmtree(img_cut_dir)
    shutil.rmtree(img_cut_mask_dir)
    shutil.rmtree(out_data_image_crop)
    


    return out_dir_data_final


    

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        '--image_dir',
        help='Orginal Image Directory',
        required=True
    )


    args_parser.add_argument(
        '--box_dir',
        help='Box cut directory',
        required=True
    )

    args_parser.add_argument(
        '--label_dir',
        help='Box cut directory',
        required=True
    )
    args_parser.add_argument(
        '--out_dir',
        help='Data output directory',
        required=True
    )
    param = args_parser.parse_args()
    
    image_dir = str(param.image_dir)
    box_dir = str(param.box_dir)
    label_dir = str(param.label_dir)
    out_dir = str(param.out_dir)
    main(image_dir,box_dir,label_dir,out_dir)    