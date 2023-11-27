import os
import argparse
import predict_ds_image_tf_model
import uuid
from params import INPUT_PATH, OUTPUT_PATH, ROOT_DATA_FOLDER
import shutil
if __name__ == '__main__':
    input_path = INPUT_PATH
    output_path = OUTPUT_PATH
    model_path = f'{ROOT_DATA_FOLDER}/pretrain-models/super_resolution_x4/v1/weight/real_esr4xkeras.h5'
    predict_ds_image_tf_model.run_sr(model_path,input_path, output_path)

