import os
import argparse
import predict_ds_image_tf_model_v2
import uuid
from params import INPUT_PATH, OUTPUT_PATH, ROOT_DATA_FOLDER,TMP_PATH
import shutil
if __name__ == '__main__':
    input_path = INPUT_PATH
    output_path = OUTPUT_PATH
    tmp_path = TMP_PATH
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    model_path = f'{ROOT_DATA_FOLDER}/pretrain-models/super_resolution_x10_sen2/v1/weight/model4'
    model_path2 = f'{ROOT_DATA_FOLDER}/pretrain-models/super_resolution_x10_sen2/v1/weight/model.h5'
    predict_ds_image_tf_model_v2.run_sr(model_path2,input_path, output_path,scale=10)
    