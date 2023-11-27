import params
from smg_imageaglignment import image_align
from predict import predict_big
import os

if __name__ == '__main__':
    print('Start running model')
    model_path = f'{params.ROOT_DATA_FOLDER}/pretrain-models/change-detect-khalifasat/v1/weights/model.pth'
    input_path_1 = params.INPUT_PATH_1
    input_path_2 = params.INPUT_PATH_2
    output_path =params.OUTPUT_PATH
    tmp_path = params.TMP_PATH
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    base_tmp = os.path.join(tmp_path,"base.tif")
    image_tmp = os.path.join(tmp_path,"image.tif")
    image_align(input_path_1,input_path_2,tmp_path)
    predict_big(base_tmp,image_tmp,model_path,tmp_path,output_path)

    

    