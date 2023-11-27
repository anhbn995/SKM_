import glob, os
import numpy as np

def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.tif"):
        list_id.append(file[:-4])
    return list_id


def coppy_shape(name_shape, source_shp_path, dest_shp_path):
    for file_ in os.listdir(source_shp_path):
        if file_.startswith(name_shape):
            os.rename(os.path.join(source_shp_path, file_), os.path.join(dest_shp_path,  file_))
    

def split_train_val(dir_img_cut_size, dir_mask_cut_size, dir_shape_label_cut_size, split, out_dir_data_split):
    image_list = create_list_id(dir_img_cut_size)
    np.random.shuffle(image_list)
    count = len(image_list)
    cut_idx = int(round(count*split))
    print(cut_idx)
    train_list = image_list[0:cut_idx]
    other_list = [id_image for id_image in image_list if id_image not in train_list]
    val_list = other_list
    # test_list = []
    path_train = os.path.join(out_dir_data_split,'train','images')
    path_train_mask = os.path.join(out_dir_data_split,'train','mask')
    path_train_label_shp = os.path.join(out_dir_data_split,'train','label_shape')
    os.makedirs(path_train, exist_ok=True)
    os.makedirs(path_train_mask, exist_ok=True)
    os.makedirs(path_train_label_shp, exist_ok=True)
    path_val = os.path.join(out_dir_data_split,'val','images')
    path_val_mask = os.path.join(out_dir_data_split,'val','mask')
    path_val_label_shp = os.path.join(out_dir_data_split,'val','label_shape')
    os.makedirs(path_val, exist_ok=True)
    os.makedirs(path_val_mask, exist_ok=True)
    os.makedirs(path_val_label_shp, exist_ok=True)
    # path_test = os.path.join(out_dir_data_split,'test','images')
    # path_test_mask = os.path.join(out_dir_data_split,'test','mask')
    # os.makedirs(path_test, exist_ok=True)
    # os.makedirs(path_test_mask, exist_ok=True)
    for image_name in train_list:
        os.rename(os.path.join(dir_img_cut_size,image_name+'.tif'), os.path.join(path_train,image_name+'.tif'))
        os.rename(os.path.join(dir_mask_cut_size,image_name+'.tif'), os.path.join(path_train_mask,image_name+'.tif'))
        coppy_shape(image_name, dir_shape_label_cut_size, path_train_label_shp)
    for image_name in val_list:
        os.rename(os.path.join(dir_img_cut_size,image_name+'.tif'), os.path.join(path_val,image_name+'.tif'))
        os.rename(os.path.join(dir_mask_cut_size,image_name+'.tif'), os.path.join(path_val_mask,image_name+'.tif'))
        coppy_shape(image_name, dir_shape_label_cut_size, path_val_label_shp)
    # for image_name in test_list:
    #     os.rename(os.path.join(dir_img_cut_size,image_name+'.tif'), os.path.join(path_test,image_name+'.tif'))
    #     os.rename(os.path.join(dir_mask_cut_size,image_name+'.tif'), os.path.join(path_test_mask,image_name+'.tif'))
    return path_train, path_train_label_shp, path_val, path_val_label_shp

if __name__=='__main__':
    dir_img_cut_size = r'E:\TMP_XOA\mongkos_std\test\data_23_06_2020\image'
    dir_mask = r'E:\TMP_XOA\mongkos_std\test\data_23_06_2020\label'
    dir_shape_label_cut_size = r''
    split = 0.8
    out_dir_data_split = r'E:\TMP_XOA\mongkos_std\test\data_23_06_2020\tmp'
    split_train_val(dir_img_cut_size, dir_mask, dir_shape_label_cut_size, split, out_dir_data_split)
