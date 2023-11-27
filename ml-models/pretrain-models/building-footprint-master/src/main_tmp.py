import os, glob
import argparse
import predict_box,predict_mask
import uuid
from tqdm import tqdm
from params import INPUT_PATH, OUTPUT_PATH, ROOT_DATA_FOLDER, TMP_PATH
import shutil
if __name__ == '__main__':
    input_dir = INPUT_PATH
    output_dir = OUTPUT_PATH
    list_input_path = glob.glob(os.path.join(input_dir, '*.tif'))
    for input_path  in tqdm(list_input_path):
        output_path = os.path.join(output_dir, os.path.basename(input_path).replace('.tif', '.shp'))
        tmp_path = f'{TMP_PATH}/{uuid.uuid4().hex}'
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        shp_tmp = os.path.join(tmp_path,"box.shp")
        model_box = f'{ROOT_DATA_FOLDER}/pretrain-models/building-footprint-master/v1/weight/model_box.h5'
        model_mask = f'{ROOT_DATA_FOLDER}/pretrain-models/building-footprint-master/v1/weight/model_mask.h5'
        predict_box.predict_main(input_path, model_box, shp_tmp, bound_path=None, out_type="bbox",verbose=1)
        if os.path.exists(shp_tmp):
            predict_mask.predict_building(input_path,shp_tmp,output_path,model_mask)
        shutil.rmtree(tmp_path)
    
