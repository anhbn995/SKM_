# from tensorflow import keras
import tensorflow as tf
import pickle

from params import INPUT_PATH, OUTPUT_PATH, ROOT_DATA_FOLDER, TMP_PATH
from .predict import *
from .utils import write_image, write_color_img
from .post_process import *

def main(input_path, output_path, tmp_path, weight_path, json_path= None):

     multiclass_weight = f'{weight_path}/monitor_5class.h5'
     multiclass_model = tf.keras.models.load_model(multiclass_weight, compile=False)
     multiclass_data = inference(multiclass_model, input_path)
     multiclass_tmp = f'{tmp_path}/multiclass.tif'
     write_image(input_path, multiclass_data, multiclass_tmp)
     mask_green = get_green_mask(multiclass_tmp)
     mask_water = get_water_mask(multiclass_tmp)

     mangrove_weight = f'{weight_path}/mangrove.h5'
     mangrove_model = tf.keras.models.load_model(mangrove_weight, compile=False)
     mangrove_data = inference(mangrove_model, input_path)
     mangrove_tmp = f'{tmp_path}/mangrove.tif'
     write_image(input_path, mangrove_data, mangrove_tmp)
     mask_mangrove = get_mangrove_mask(mangrove_tmp)


     grass_built_weight = f'{weight_path}/grass_builtup.mdl'
     with open(grass_built_weight, 'rb') as f:
          data = pickle.load(f)
     
     grass_built_model = data['model']
     grass_built_data = predict_pixel_model(grass_built_model, input_path, crop_size=512)
     grass_built_tmp = f'{tmp_path}/grass_built.tif'
     write_image(input_path, grass_built_data, grass_built_tmp)
     mask_grass = get_grass_mask(grass_built_tmp)
     mask_builtup = get_builtup_mask(grass_built_tmp)


     final_result = gen_final_result(input_path, mask_green, mask_grass, mask_mangrove, mask_water, mask_builtup)
     write_color_img(multiclass_tmp, final_result, output_path)

     return True

if __name__ == "__main__":
     input_path = INPUT_PATH
     output_path = OUTPUT_PATH
     tmp_path = TMP_PATH
     weight_path = f'{ROOT_DATA_FOLDER}/pretrain-models/panama_class/v1/weights'
     json_path = f'{ROOT_DATA_FOLDER}/pretrain-models/panama_class/v1/model.json'
     main(input_path, output_path, tmp_path, weight_path, json_path)
     print("Finished")
     import sys
     sys.exit()
