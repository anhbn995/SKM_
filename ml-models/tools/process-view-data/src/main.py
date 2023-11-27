import os, glob
from ultils import *
from clip_shape import clip_all_shape
from clip_size_img import main_gen_data_with_size
from split_train_val import split_train_val
from gen_anotation import main_gen_anotation
from params import INPUT_IMAGES_DIR,INPUT_LABEL_PATH,SAMPLE_SIZE,GEN_THEM,SPLIT,TMP_PATH,OUTPUT_DIR
from task import task_proccessing_percent
def get_fn_in_dir_image(dir_img):
    return [os.path.basename(fp).replace('.tif','')  for fp in glob.glob(os.path.join(dir_img,'*.tif'))]


def create_mask_by_img_aoi_clip(dir_img_clip, fp_shp_label, out_mask_dir):
    list_fname = get_fn_in_dir_image(dir_img_clip)
       
    for fname in list_fname:
       fp_image = os.path.join(dir_img_clip, fname + '.tif')
       fp_mask_out = os.path.join(out_mask_dir, fname + '.tif')
       create_mask_by_shape(fp_shp_label, fp_image, fp_mask_out)
    print('Done gen AOI mask')



def gen_dataset_training_fastRCNN(dir_img_aoi, fp_shp_label, out_tmp, sampleSize, gen_them, split, out_dir_data_split):
    dir_mask_aoi = os.path.join(out_tmp,'mask_aoi')
    if sampleSize:
        outputFolder_cut_size = os.path.join(out_tmp, f'cut_size_{sampleSize}')
    else:
        outputFolder_cut_size = os.path.join(out_tmp, 'cut_size')
    os.makedirs(dir_mask_aoi, exist_ok=True)
    os.makedirs(outputFolder_cut_size, exist_ok=True)
    
    # create Mask 
    task_proccessing_percent(0.1)
    create_mask_by_img_aoi_clip(dir_img_aoi, fp_shp_label, dir_mask_aoi)
    task_proccessing_percent(0.2)
    # cut size img, mask, label
    dir_img_cut_size, dir_mask_cut_size = main_gen_data_with_size(dir_img_aoi, dir_mask_aoi, outputFolder_cut_size, sampleSize=sampleSize, gen_them=gen_them)
    
    dir_out_shape_cut_size = os.path.join(outputFolder_cut_size, 'shape_label')
    clip_all_shape(dir_img_cut_size, fp_shp_label, dir_out_shape_cut_size)
    
    # split train/val
    path_train_img, path_train_label_shp, path_val_img, path_val_label_shp = split_train_val(dir_img_cut_size, dir_mask_cut_size, dir_out_shape_cut_size, split, out_dir_data_split)
    
    # gen annotation
    main_gen_anotation(path_train_img, path_train_label_shp)
    main_gen_anotation(path_val_img, path_val_label_shp)


def merge_dataset_training_fastRCNN(list_traning_dataset, dest_new_trainingdataset_dir):
    move_list_dir_to_dst_dir(list_traning_dataset, dest_new_trainingdataset_dir)
    path_train_img = os.path.join(dest_new_trainingdataset_dir, 'train', 'images')
    path_train_label_shp = os.path.join(dest_new_trainingdataset_dir, 'train', 'label_shape')
    path_val_img = os.path.join(dest_new_trainingdataset_dir, 'val', 'images')
    path_val_label_shp = os.path.join(dest_new_trainingdataset_dir, 'val', 'label_shape')
    main_gen_anotation(path_train_img, path_train_label_shp)
    main_gen_anotation(path_val_img, path_val_label_shp)

    
if __name__=='__main__':
    gen_dataset_training_fastRCNN(INPUT_IMAGES_DIR, INPUT_LABEL_PATH, TMP_PATH, SAMPLE_SIZE, GEN_THEM, SPLIT, OUTPUT_DIR)

# docker run -it -v /home/geoai/geoai_data_test2/ad/training_data/:/data -e INPUT_IMAGES_DIR=/data/img_cut -e INPUT_LABEL_PATH=/data/label/label.shp -e TMP_PATH=/data/tmp -e OUTPUT_DIR=/data/output process-view-data python src/main.py