import params
from predict import OscarPalmTreecounting
if __name__ == '__main__':
    model_path = f'{params.ROOT_DATA_FOLDER}/pretrain-models/tree-counting-oscar-palm/v1/weights/model.h5'
    input_path = params.INPUT_PATH
    tmp_path = params.TMP_PATH
    output_path = params.OUTPUT_PATH
    OscarPalmTreecounting().main(input_path, model_path, tmp_path, output_path)
