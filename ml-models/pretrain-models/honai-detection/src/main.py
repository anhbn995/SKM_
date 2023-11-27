import os
import argparse
import predict_box
import uuid
from params import INPUT_PATH, OUTPUT_PATH, ROOT_DATA_FOLDER
import shutil
if __name__ == '__main__':
    input_path = INPUT_PATH
    output_path = OUTPUT_PATH
    model_box = f'{ROOT_DATA_FOLDER}/pretrain-models/honai-detection/v1/weight/model_box.h5'
    predict_box.predict_main(input_path, model_box, output_path, bound_path=None, out_type="bbox",verbose=1)
    
