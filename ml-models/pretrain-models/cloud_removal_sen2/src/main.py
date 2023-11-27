import tensorflow as tf
from tensorflow.keras.models import load_model

from params import INPUT_PATH_1, INPUT_PATH_2, OUTPUT_PATH, ROOT_DATA_FOLDER, TMP_PATH
from .predict import *

if __name__ == "__main__":
     input_path_s1 = INPUT_PATH_2
     input_path_s2_cloudy = INPUT_PATH_1
     output_path = OUTPUT_PATH
     tmp_path = TMP_PATH
     weight_path = f'{ROOT_DATA_FOLDER}/pretrain-models/cloud_removal_sen2/v1/weights'
     json_path = f'{ROOT_DATA_FOLDER}/pretrain-models/cloud_removal_sen2/v1/model.json'

     # cloudy_img_path = '/home/boom/data/ml-models/storage/pretrain-models/cloud_removal_sen2/v1/input_data/s2_cloudy.tif'
     # sar_img_path = '/home/boom/data/ml-models/storage/pretrain-models/cloud_removal_sen2/v1/input_data/s1.tif'

     model_path = f'{weight_path}/cloud_removal.h5'
     model = load_model(model_path, compile=False, custom_objects={'tf': tf})

     predict(model, input_path_s2_cloudy, input_path_s1, output_path, size=256)
     print("Finished")
     import sys
     sys.exit()
