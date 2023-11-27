from environs import Env
env = Env()
env.read_env()
import json
DATA_DIR = env.str('DATA_DIR')
LOG_DIR = env.str('LOG_DIR')
RPN_ANCHOR_SCALES = tuple(json.loads(env.str('RPN_ANCHOR_SCALES'))) #[16,32,64,128,256]
BACKBONE=env.str('BACKBONE', 'resnet101')
PRETRAIN_MODEL = env.str('PRETRAIN_MODEL') or None
MODEL_NAME=env.str('MODEL_NAME')
MODEL_SIZE = env.int('MODEL_SIZE')
STAGE_1 = env.int('STAGE_1', 0)
STAGE_2 = env.int('STAGE_2', 1)
STAGE_3 = env.int('STAGE_3', 1)
TRAIN_STEP = env.int('TRAIN_STEP', 500)
VAL_STEP = env.int('VAL_STEP', 200)
ROI_POSITIVE_RATIO = env.float('ROI_POSITIVE_RATIO', 0.33)
RPN_NMS_TH = env.float('RPN_NMS_TH', 0.7)
MAX_GT_INSTANCES = env.int('MAX_GT_INSTANCES', 100)
HOSTED_ENDPOINT = env.str('HOSTED_ENDPOINT','')
TRAINING_ID = env.str('TRAINING_ID','')
PROCESS_ID = env.str('PROCESS_ID','')




# Please make sure every model must have 4 input below
# This is root folder where related data stored. You must name your folder in root folder as same as your model folder name
# Example: ROOT_DATA_FOLDER/type-of-model/model-name
# ROOT_DATA_FOLDER = env.str('ROOT_DATA_FOLDER')
