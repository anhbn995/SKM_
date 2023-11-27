import params
from predict import main_predict
if __name__ == '__main__':
    model_path = f'{params.ROOT_DATA_FOLDER}/pretrain-models/building-footprint/v1/weights/model.h5'
    model_dir = params.TMP_PATH
    input_path = params.INPUT_PATH
    ouput_path = params.OUTPUT_PATH
    main_predict(input_path, model_path, model_dir, ouput_path)
