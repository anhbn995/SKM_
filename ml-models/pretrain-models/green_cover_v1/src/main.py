# from tensorflow import keras
import tensorflow as tf
import pickle

from params import INPUT_PATH, OUTPUT_PATH, ROOT_DATA_FOLDER, TMP_PATH
from .predict import *
from .utils import write_color_img

def main(input_path, output_path, tmp_path, weight_path, json_path= None):

     tmp_result = f'{tmp_path}/green_tmp.tif'
     green_weight = f'{weight_path}/green_cover.h5'
     green_model = tf.keras.models.load_model(green_weight, compile=False)
     predict_farm(green_model, input_path, tmp_result, 128)
     write_color_img(tmp_result, output_path)

     return True

if __name__ == "__main__":
     input_path = INPUT_PATH
     output_path = OUTPUT_PATH
     tmp_path = TMP_PATH
     weight_path = f'{ROOT_DATA_FOLDER}/pretrain-models/green_cover_v1/v1/weights'
     json_path = f'{ROOT_DATA_FOLDER}/pretrain-models/green_cover_v1/v1/model.json'
     main(input_path, output_path, tmp_path, weight_path, json_path)
     print("Finished")
     import sys
     sys.exit()
