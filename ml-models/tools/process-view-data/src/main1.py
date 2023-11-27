from ultils import *
from gen_anotation import main_gen_anotation
from param1s import TRAINING_DATASET_DIRS, OUTPUT_DIR
def merge_dataset_training_fastRCNN(list_traning_dataset, dest_new_trainingdataset_dir):
    move_list_dir_to_dst_dir(list_traning_dataset, dest_new_trainingdataset_dir)
    path_train_img = os.path.join(dest_new_trainingdataset_dir, 'train', 'images')
    path_train_label_shp = os.path.join(dest_new_trainingdataset_dir, 'train', 'label_shape')
    path_val_img = os.path.join(dest_new_trainingdataset_dir, 'val', 'images')
    path_val_label_shp = os.path.join(dest_new_trainingdataset_dir, 'val', 'label_shape')
    main_gen_anotation(path_train_img, path_train_label_shp)
    main_gen_anotation(path_val_img, path_val_label_shp)

if __name__=='__main__':
    merge_dataset_training_fastRCNN(TRAINING_DATASET_DIRS, OUTPUT_DIR)