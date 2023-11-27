import params
from predict import CarCountingCombine2Stride
if __name__ == '__main__':
    model_path = f'{params.ROOT_DATA_FOLDER}/pretrain-models/car-counting-nunukan/v1/weights/model.h5'
    input_path = params.INPUT_PATH
    tmp_path = params.TMP_PATH
    output_path = params.OUTPUT_PATH
    CarCountingCombine2Stride().main(input_path, model_path, tmp_path, output_path)
