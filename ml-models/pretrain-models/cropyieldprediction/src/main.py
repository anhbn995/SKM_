import os
import argparse
import uuid
from params import INPUT_PATH1,INPUT_PATH2,INPUT_PATH3,INPUT_PATH4, OUTPUT_PATH, ROOT_DATA_FOLDER,TMP_PATH
from imagealign import align
import shutil
import uuid
from predict import predict,loadmodel
from stretch import stretch
from osgeo import gdal
if __name__ == '__main__':
    model_path1= f'{ROOT_DATA_FOLDER}/pretrain-models/cropyieldprediction/v1/weight/S-S2.h5'
    model_path2= f'{ROOT_DATA_FOLDER}/pretrain-models/cropyieldprediction/v1/weight/SP-S2.h5'
    model_path3= f'{ROOT_DATA_FOLDER}/pretrain-models/cropyieldprediction/v1/weight/SPE-S2.h5'
    tmp_path = f'{TMP_PATH}/{uuid.uuid4().hex}'
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    TMP_PATH1 = os.path.join(tmp_path,"tmp1.tif")
    TMP_PATH2 = os.path.join(tmp_path,"tmp2.tif")
    TMP_PATH3 = os.path.join(tmp_path,"tmp3.tif")
    IMG_TMP_PATH1 = os.path.join(tmp_path,"input1.tif")
    IMG_TMP_PATH2 = os.path.join(tmp_path,"input2.tif")
    IMG_TMP_PATH3 = os.path.join(tmp_path,"input3.tif")
    
    if INPUT_PATH4 != "None":
        print(1)
        gdal.Warp(TMP_PATH1,
                INPUT_PATH1,
                dstSRS='EPSG:4326',
                cutlineDSName=INPUT_PATH4,
                cropToCutline = True)
        if INPUT_PATH2 != "None":
            gdal.Warp(TMP_PATH2,
                    INPUT_PATH2,
                    dstSRS='EPSG:4326',
                    cutlineDSName=INPUT_PATH4,
                    cropToCutline = True)
        if INPUT_PATH3 != "None":
            gdal.Warp(TMP_PATH3,
                    INPUT_PATH3,
                    dstSRS='EPSG:4326',
                    cutlineDSName=INPUT_PATH4,
                    cropToCutline = True)
    
    else:
        TMP_PATH1=INPUT_PATH1
        TMP_PATH2=INPUT_PATH2
        TMP_PATH3=INPUT_PATH3
    shutil.copy(TMP_PATH1,IMG_TMP_PATH1)

    if INPUT_PATH2 != "None":
        align(TMP_PATH2,IMG_TMP_PATH1,IMG_TMP_PATH2)
    else:
        INPUT_PATH2 = None
    if INPUT_PATH3 != "None":
        align(TMP_PATH3,IMG_TMP_PATH1,IMG_TMP_PATH3)
    else:
        INPUT_PATH3 = None

    if INPUT_PATH2 and INPUT_PATH3:
        input_paths=[IMG_TMP_PATH1,IMG_TMP_PATH2,IMG_TMP_PATH3]
        model_path = model_path3
    elif INPUT_PATH2 and not(INPUT_PATH3):
        input_paths=[IMG_TMP_PATH1,IMG_TMP_PATH2]
        model_path = model_path2
    else:
        input_paths=[IMG_TMP_PATH1]
        model_path = model_path1
    model=loadmodel(model_path)

    # input_paths = INPUT_PATH1
    output_path = OUTPUT_PATH
    inputfiles_tmp=[f.replace(".tif", "_tmp.tif") for f in input_paths]
    for i in range(len(input_paths)):
        stretch(input_paths[i], inputfiles_tmp[i])
    predict(inputfiles_tmp, output_path, model)
    # model_path = f'{ROOT_DATA_FOLDER}/pretrain-models/'
    # shutil.copy(input_paths,output_path)

