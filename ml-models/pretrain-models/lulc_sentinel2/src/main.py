# from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import load_model

from params import INPUT_PATH, OUTPUT_PATH, ROOT_DATA_FOLDER, TMP_PATH
from .predict import *

if __name__ == "__main__":
     input_path = INPUT_PATH
     output_path = OUTPUT_PATH
     tmp_path = TMP_PATH
     weight_path = f'{ROOT_DATA_FOLDER}/pretrain-models/lulc_sentinel2/v1/weights'
     json_path = f'{ROOT_DATA_FOLDER}/pretrain-models/lulc_sentinel2/v1/model.json'

     # image_path = '/home/skymap/BoVu/LULC/images_6band/test.tif'
     # output_tif = '/home/skymap/BoVu/LULC/tmp/test.tif'
     # weight_path = '/home/skymap/BoVu/LULC/models_LULC_indo'

     green_model = load_model(f'{weight_path}/green.h5', compile=False)
     bareland_model = load_model(f'{weight_path}/bareland.h5', compile=False)
     crops_model = load_model(f'{weight_path}/crops.h5', compile=False)
     forest_model = load_model(f'{weight_path}/forest.h5', compile=False)
     builtup_model = load_model(f'{weight_path}/builtup.h5', compile=False)
     water_model = load_model(f'{weight_path}/water.h5', compile=False)
     list_models = [bareland_model,crops_model,forest_model,builtup_model,water_model,green_model]
     predict(input_path, output_path, size=128, list_models=list_models, numbands=6)
     print("Finished")
     import sys
     sys.exit()
