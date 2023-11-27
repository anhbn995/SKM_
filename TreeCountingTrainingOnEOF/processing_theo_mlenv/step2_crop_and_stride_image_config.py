import datetime
now = datetime.datetime.now()
dt = now.strftime("%Y%m%d_%H%M%S")


"""Grass CAOHOC"""
crop_size = 512
stride_size = 256
dir_img_and_mask = r'/home/skm/SKM16/A_CAOHOC/Build_data_train_uint8_cua_3Class/Grass'
name_folder_img_and_mask = ["img_unit8_cut_box_Grass", "img_unit8_cut_box_Grass_mask"]
rename_folder = ['images', 'masks']
name_project = f"Grass_UINT8_{crop_size}_stride_size_{stride_size}"

"""Tree CAOHOC"""
crop_size = 512
stride_size = 256
dir_img_and_mask = r'/home/skm/SKM16/A_CAOHOC/Build_data_train_uint8_cua_3Class/Tree'
name_folder_img_and_mask = ["img_unit8_cut_box_Tree", "img_unit8_cut_box_Tree_mask"]
rename_folder = ['image', 'label']
name_project = f"Tree_UINT8_{crop_size}_stride_size_{stride_size}"

"""Water CAOHOC"""
crop_size = 512
stride_size = 256
dir_img_and_mask = r'/home/skm/SKM16/A_CAOHOC/Build_data_train_uint8_cua_3Class/Water'
name_folder_img_and_mask = ["img_unit8_cut_box_Water", "img_unit8_cut_box_Water_mask"]
rename_folder = ['image', 'label']
name_project = f"Water_UINT8_{crop_size}_stride_size_{stride_size}"


"""Planet AOI GreenCover"""
crop_size = 512
stride_size = 200
dir_img_and_mask = r'/home/skm/SKM16/Planet_GreenChange/Data_4band_Green'
name_folder_img_and_mask = ["ImgRGBNir_8bit_perimage_cut", "ImgRGBNir_8bit_perimage_cut_mask"]
rename_folder = ['image', 'label']
name_project = f"Green_UINT8_4band_cut_{crop_size}_stride_{stride_size}"


"""Planet AOI GreenCover Miner """
crop_size = 512
stride_size = 200
dir_img_and_mask = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/planet'
name_folder_img_and_mask = ["images_8bit_perimage_cut", "images_8bit_perimage_cut_mask"]
rename_folder = ['images', 'masks']
name_project = f"Green_UINT8_4band_cut_{crop_size}_stride_{stride_size}"


"""Planet AOI GreenCover Miner """
crop_size = 512
stride_size = 200
dir_img_and_mask = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/TrainingDataSet/Water'
name_folder_img_and_mask = ["img_ori_8bit_perimage_cut", "img_ori_8bit_perimage_cut_mask"]
rename_folder = ['images', 'masks']
name_project = f"Water_UINT8_4band_cut_{crop_size}_stride_{stride_size}"


"""Planet AOI GreenCover V2 Miner _cai nay verry good"""
# crop_size = 512
# stride_size = 200
# dir_img_and_mask = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/planet/V2'
# name_folder_img_and_mask = ["images_8bit_perimage_cut", "images_8bit_perimage_cut_mask"]
# rename_folder = ['images', 'masks']
# name_project = f"Water_UINT8_4band_cut_{crop_size}_stride_{stride_size}"





"""Truot Lo"""
name_object = 'Truot_Lo'
crop_size = 64
stride_size = 40
numband = 4
type_data = 'Uint8'
dir_img_and_mask = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage'
name_folder_img_and_mask = ["Img_uint8_crop", "mask_img_sen2_TruotLo"]
rename_folder = ['images', 'masks']
name_project = f"{name_object}_{type_data}_{numband}band_cut_{crop_size}_stride_{stride_size}_time_{dt}"


"""Planet Cloud for all"""
name_object = 'Cloud'
type_data = 'Original' # 'Uint8' or 'Original'
numband = 4
crop_size = 512
stride_size = 200
dir_img_and_mask = r'/home/skm/SKM16/Data/Planet/Cloud_planet/cloud_iou/Training_dataset/V1'
name_folder_img_and_mask = ["Img_original_2468_BGRNir_cut", "Img_original_2468_BGRNir_cut_mask"]
rename_folder = ['images', 'masks']
name_project = f"{name_object}_{type_data}_{numband}band_cut_{crop_size}_stride_{stride_size}"


name_object = 'Cloud'
type_data = 'Uint8' # 'Uint8' or 'Original'
numband = 4
crop_size = 512
stride_size = 200
dir_img_and_mask = r'/home/skm/SKM16/Data/Planet/Cloud_planet/cloud_iou/Training_dataset/V2'
name_folder_img_and_mask = ["Img_original_2468_BGRNir_8bit_perimage_cut", "Img_original_2468_BGRNir_8bit_perimage_cut_mask"]
rename_folder = ['images', 'masks']
name_project = f"{name_object}_{type_data}_{numband}band_cut_{crop_size}_stride_{stride_size}_time_{dt}"







name_object = 'ALL_Water'
type_data = 'Original' # 'Uint8' or 'Original'
numband = 4
crop_size = 512
stride_size = 200
# v1
# dir_img_and_mask = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/training_with_original'
# name_folder_img_and_mask = ["img_ori_cut", "img_ori_cut_mask"]
# v2
dir_img_and_mask = r'/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/DataTraining_Origin/Water'
name_folder_img_and_mask = ["img_ori_cut", "img_ori_cut_mask"]
rename_folder = ['images', 'masks']
name_project = f"{name_object}_{type_data}_{numband}band_cut_{crop_size}_stride_{stride_size}_time_{dt}"



name_object = 'ALL_Green'
type_data = 'Original' # 'Uint8' or 'Original'
numband = 4
crop_size = 512
stride_size = 200
dir_img_and_mask = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/DataTraining_Origin/Green"
name_folder_img_and_mask = ["img_ori_cut_V2", "img_ori_cut_V2_mask"]
rename_folder = ['images', 'masks']
name_project = f"{name_object}_{type_data}_{numband}band_cut_{crop_size}_stride_{stride_size}_time_{dt}"

# name_object = 'Green_original'
# type_data = 'Original' # 'Uint8' or 'Original'
# numband = 4
# crop_size = 512
# stride_size = 200
# dir_img_and_mask = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/training_with_original/green_trainig'
# name_folder_img_and_mask = ["img_ori_cut", "img_ori_cut_mask"]
# rename_folder = ['images', 'masks']
# name_project = f"{name_object}_{type_data}_{numband}band_cut_{crop_size}_stride_{stride_size}_time_{dt}"

name_object = 'TauBien'
type_data = 'Original' # 'Uint8' or 'Original'
numband = 3
crop_size = 128
stride_size = 64
dir_img_and_mask = r"/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/data_faster-rnn/image_crop_box/tmp"
name_folder_img_and_mask = ["TauBien_cut_img", "TauBien_cut_img_mask"]
rename_folder = ['images', 'masks']
name_project = f"{name_object}_{type_data}_{numband}band_cut_{crop_size}_stride_{stride_size}_time_{dt}"
