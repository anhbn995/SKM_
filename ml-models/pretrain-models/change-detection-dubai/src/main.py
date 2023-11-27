import params
from predict_change_dubai import main
if __name__ == '__main__':
    model_path = f'{params.ROOT_DATA_FOLDER}/pretrain-models/change-detection-dubai/v1/weights/u2net_change_dubai.h5'
    input_path = params.INPUT_PATH
    output_path = params.OUTPUT_PATH
    main(input_path, output_path, model_path)
