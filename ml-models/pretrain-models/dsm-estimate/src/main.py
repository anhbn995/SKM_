import params
from predict_height import predict_height
import os

if __name__ == '__main__':
    print('Start running model')
    model_path = f'{params.ROOT_DATA_FOLDER}/pretrain-models/dsm-estimate/v1/weights/model.pth.tar'
    input_path = params.INPUT_PATH
    output_path =params.OUTPUT_PATH
    tmp_path = params.TMP_PATH
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    predict_height(input_path, output_path, model_path)
