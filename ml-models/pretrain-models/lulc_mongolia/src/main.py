from unittest import result
import uuid
import glob
from keras.models import load_model
from params import INPUT_PATH, OUTPUT_PATH, ROOT_DATA_FOLDER, TMP_PATH
from .predict import *

def main(input_path, output_path, tmp_path, weight_path, json_path= None):

     crop_size = 1000
     
     nobuilding_weight = f'{weight_path}/nobuilding.h5'
     nobuilding_model = load_model(nobuilding_weight, compile=False)
     result_nobuilding = f'{tmp_path}/{uuid.uuid4().hex}.tif'
     predict_nobuilding(result_nobuilding, input_path, crop_size, nobuilding_model)

     building_weight = f'{weight_path}/building.h5'
     buiding_model = load_model(building_weight)
     result_building = f'{tmp_path}/{uuid.uuid4().hex}.tif'
     predict_building(input_path, buiding_model, result_building)

     add_building(result_nobuilding, result_building)
     morpho_result(result_nobuilding, output_path)
     return True

if __name__ == "__main__":
     input_path = INPUT_PATH
     output_path = OUTPUT_PATH
     tmp_path = TMP_PATH
     weight_path = f'{ROOT_DATA_FOLDER}/pretrain-models/lulc_mongolia/v1/weights'
     json_path = f'{ROOT_DATA_FOLDER}/pretrain-models/lulc_mongolia/v1/model.json'
     main(input_path, output_path, tmp_path, weight_path, json_path)
     print("Finished")
     import sys
     sys.exit()
