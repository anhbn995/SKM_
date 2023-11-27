import os, glob
from ultils import *
from clip_shape import clip_all_shape
from clip_size_img import main_gen_data_with_size
from split_train_val import split_train_val
from gen_anotation import main_gen_anotation


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
    create_mask_by_img_aoi_clip(dir_img_aoi, fp_shp_label, dir_mask_aoi)
    # cut size img, mask
    dir_img_cut_size, dir_mask_cut_size = main_gen_data_with_size(dir_img_aoi, dir_mask_aoi, outputFolder_cut_size, sampleSize=sampleSize, gen_them=gen_them)
    
    # cut size label
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
    dir_img_aoi = r'/home/skm/SKM16/Tmp/TreeCounting/Test_EOF/TrainingDataSet/img_aoi'
    fp_shp_label = r"/home/skm/SKM16/Tmp/TreeCounting/Test_EOF/TrainingDataSet/label/labl.shp"
    out_tmp = r'/home/skm/SKM16/Tmp/TreeCounting/Test_EOF/TrainingDataSet/tmp_xoa'
    sampleSize = None
    gen_them = False
    split = 0.8
    out_dir_data_split = r'/home/skm/SKM16/Tmp/TreeCounting/Test_EOF/TrainingDataSet/traning_fasterRcnn1'
    gen_dataset_training_fastRCNN(dir_img_aoi, fp_shp_label, out_tmp, sampleSize, gen_them, split, out_dir_data_split)
    
    # source_folders = [r'E:\TMP_XOA\mongkos_std\tmp_xoa\traning_fasterRcnn1', r'E:\TMP_XOA\mongkos_std\tmp_xoa\traning_fasterRcnn2']
    # dst_dir =  r'E:\TMP_XOA\mongkos_std\tmp_xoa\traning_fasterRcnn_zz'
    # merge_dataset_training_fastRCNN(source_folders, dst_dir)