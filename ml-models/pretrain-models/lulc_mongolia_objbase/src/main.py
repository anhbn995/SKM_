from unittest import result
import uuid
from keras.models import load_model
from params import INPUT_PATH, OUTPUT_PATH, ROOT_DATA_FOLDER, TMP_PATH
from .predict import *
from .slic import *
from .utils import *
import os
import shutil

def main(input_path, output_path, tmp_path, weight_path, json_path= None):
     name_ = uuid.uuid4().hex
     temp_folder = f'{tmp_path}/{name_}'
     if not os.path.exists(temp_folder):
          os.makedirs(temp_folder)
     out_slic_tif = f'{temp_folder}/{name_}_slic.tif'
     out_slic_shp = f'{temp_folder}/{name_}_slic.shp'

     qt_scheme = get_quantile_schema(input_path)
     slic_path = slic_image(input_path, out_slic_tif, qt_scheme)
     print('slic done ...')
     polygonize(slic_path, out_slic_shp)
     
     model = load_model(weight_path, compile=False)
     df_shp  = predict_to_shp(model, out_slic_shp, input_path, qt_scheme)
     print('predict done ......')
     df_to_tif(input_path, df_shp, output_path)

     try:
          shutil.rmtree(temp_folder)
     except:
          pass
     return True

if __name__ == "__main__":
     input_path = INPUT_PATH
     output_path = OUTPUT_PATH
     tmp_path = TMP_PATH
     weight_path = f'{ROOT_DATA_FOLDER}/pretrain-models/lulc_mongolia_objbase/v1/weights/mongolia.h5'
     json_path = f'{ROOT_DATA_FOLDER}/pretrain-models/lulc_mongolia_objbase/v1/model.json'

     # input_path = '/home/boom/boom/SLIC_DBSCAN_Superpixels/examples/images/test.tif'
     # output_path = '/home/boom/boom/SLIC_DBSCAN_Superpixels/examples/hatinh.tif'
     # tmp_path = '/home/boom/boom/SLIC_DBSCAN_Superpixels/examples/tmp/temp'
     # weight_path = '/home/boom/boom/SLIC_DBSCAN_Superpixels/examples/models/mongolia.h5'

     main(input_path, output_path, tmp_path, weight_path, json_path=None)
     print("Finished")
     import sys
     sys.exit()
