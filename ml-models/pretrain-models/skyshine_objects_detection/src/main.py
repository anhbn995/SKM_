from params import INPUT_PATH, OUTPUT_PATH, ROOT_DATA_FOLDER, TMP_PATH
from .predict import *


if __name__ == "__main__":
     input_path = INPUT_PATH
     output_path = OUTPUT_PATH
     tmp_path = TMP_PATH
     weight_path = f'{ROOT_DATA_FOLDER}/pretrain-models/skyshine_objects_detection/v1/weights'
     

     model_path_chanel = f'{weight_path}/u2net_512_chanel_V1_fix_model.h5'
     model_path_pipeline = f'{weight_path}/u2net_512_Pipeline_V0_model.h5'
     model_path_pond = f'{weight_path}/u2net_512_pond_V0_model.h5'
     list_fp_model = [model_path_chanel, model_path_pipeline, model_path_pond]
     main_predict(list_fp_model, input_path, output_path, size, tmp_path)
     print("Finished")
