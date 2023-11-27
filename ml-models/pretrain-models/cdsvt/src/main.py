import os
import params
from predict import *
import uuid
# from params import INPUT_PATH, OUTPUT_PATH,ROOT_DATA_FOLDER
import shutil
import model as modellib
from config import Config
import tensorflow as tf
from tensorflow.keras.models import load_model
from pytesseract import pytesseract

# Đặt đường dẫn cho Tesseract từ biến môi trường

MODEL_DIR=""
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')

# if len(gpu_devices) > 0:
#     tf.config.experimental.set_memory_growth(gpu_devices[0], True)
# tf.compat.v1.enable_eager_execution()

class InferenceConfig(Config):
    """Config for predict tree counting model"""
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # 1 Background + 1 Building
    # IMAGE_MAX_DIM = 512+64
    # IMAGE_MIN_DIM = 512+64
    IMAGE_MAX_DIM = 512+64
    IMAGE_MIN_DIM = 512+64
    # DETECTION_MAX_INSTANCES = 500
    DETECTION_MAX_INSTANCES = 20
    # MAX_GT_INSTANCES = 60
    MAX_GT_INSTANCES = 14

    MASK_SHAPE = [28, 28]

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)

    NAME = "tree_counting_model"

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    # RPN_ANCHOR_SCALES = (8,16, 32, 64, 128)

    DETECTION_NMS_THRESHOLD = 0.3
    DETECTION_MIN_CONFIDENCE = 0.7
if __name__ == '__main__':
    # pytesseract.tesseract_cmd = os.environ.get('TESSERACT_PATH', f'{INPUT_PATH}tessdata')
    # os.environ['TESSDATA_PREFIX'] = f'{INPUT_PATH}tessdata'
    
    config = InferenceConfig()
    model_detect_location_number = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_DIR, config=config)
    # f'{INPUT_PATH}/2-1-5-1-2.pdf'
    input_path = params.INPUT_PATH 
    name2 = os.path.basename(input_path).replace('.pdf','')
    
    output_path = params.OUTPUT_PATH
    model_path_edge =  f'{params.ROOT_DATA_FOLDER}/pretrain-models/cdsvt/v1/weights/u2net_256_transcript_mrsac_v1_model.h5'
    model_path_location_number = f'{params.ROOT_DATA_FOLDER}/pretrain-models/cdsvt/v1/weights/mask_rcnn_detect_number_512_0600.h5'
    model_path_classify = f'{params.ROOT_DATA_FOLDER}/pretrain-models/cdsvt/v1/weights/model_classifi_196_0.00v1.h5'


    model_detect_edge = tf.keras.models.load_model(model_path_edge)


    config = InferenceConfig()
    model_detect_location_number = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_DIR, config=config)
    model_detect_location_number.load_weights(model_path_location_number, by_name=True)
    # model_detect_location_number.summary()
   
    model_cassify = load_model(model_path_classify)
    
    predict_main(input_path,output_path,model_detect_edge, model_detect_location_number,model_cassify,config,name2)

