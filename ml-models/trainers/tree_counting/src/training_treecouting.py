import os, re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import keras
import requests
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn.dataset import TreeCountingDataset
from params import DATA_DIR ,LOG_DIR , RPN_ANCHOR_SCALES , BACKBONE, PRETRAIN_MODEL, \
                    MODEL_NAME, MODEL_SIZE , STAGE_1 , STAGE_2 , STAGE_3 , TRAIN_STEP , VAL_STEP , \
                    ROI_POSITIVE_RATIO , RPN_NMS_TH , MAX_GT_INSTANCES, HOSTED_ENDPOINT, TRAINING_ID, PROCESS_ID


class TrainingCallback(keras.callbacks.Callback):

    def __init__(self, training_id):
        super(TrainingCallback, self).__init__()
        self.training_id = training_id

    def on_epoch_end(self, epoch, logs=None):
        if not HOSTED_ENDPOINT:
            return
        keys = list(logs.keys())
        payload = []
        for key in keys:
            payload.append({
                'key': key,
                'value': str(logs[key])
            })
        payload = {
            'process_id': PROCESS_ID,
            'epoch': {
                'id': epoch + 1,
                'payload': payload
            }
        }
        url = '{}/internal/trainings/{}'.format(HOSTED_ENDPOINT, self.training_id)
        request = requests.post(url, json=payload)
        print("Request ***********************************************", request)



def training_model(data_dir, log_dir, rpn_anchor_scales, backbone,
                   pretrain_model_path=None, model_name="tree_mask_rcnn", model_size=512,  
                   stage_1=None, stage_2=None, stage_3=None, train_step=500, val_step=100, 
                   roi_positive_ratio=0.33, rpn_nms_th=0.7, max_gt_instances=100, training_id=None):
    
    class TreeCountingConfig(Config):
        # BACKBONE = "resnet101" or "resnet50"
        BACKBONE = backbone
        NAME = model_name
        RPN_NMS_THRESHOLD = rpn_nms_th
        MAX_GT_INSTANCES = max_gt_instances
        RPN_ANCHOR_SCALES = rpn_anchor_scales
        ROI_POSITIVE_RATIO = roi_positive_ratio
        STEPS_PER_EPOCH = train_step
        VALIDATION_STEPS = val_step
        
        IMAGES_PER_GPU = 1
        GPU_COUNT = 1
        NUM_CLASSES = 1 + 1  # 1 Background + 1 Building

        IMAGE_RESIZE_MODE = "square"
        IMAGE_MAX_DIM = model_size + 64
        IMAGE_MIN_DIM = model_size - 64
        
    config = TreeCountingConfig()
    config.display()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=log_dir)
    
    # Load pretrained weights
    if pretrain_model_path:
        model_path = pretrain_model_path
        model.load_weights(model_path, by_name=True)
        # Get name and step for continue training
        fn_model = os.path.basename(pretrain_model_path)
        
        struct_file = r'^mask_rcnn_[{}a-zA-Z0-9_.+-]+_[0-9]+.h5$'
        def is_valid_email(string_):
            return re.match(struct_file, string_) is not None
        
        if is_valid_email(fn_model):
            struct_number = r'_[0-9]+.h5$'
            num_epoch = re.search(struct_number, fn_model).group(0)
            num_stage_1 = int(re.findall('\d+',num_epoch)[0])
    else:
        num_stage_1 = 0
           

    # Load training dataset
    dataset_train = TreeCountingDataset()
    dataset_train.load_dataset(dataset_dir=os.path.join(data_dir, "train"), load_small=False)
    dataset_train.prepare()

    # Load validation dataset
    dataset_val = TreeCountingDataset()
    dataset_val.load_dataset(dataset_dir=os.path.join(data_dir, "val"), load_small=False, return_coco=True)
    dataset_val.prepare()
    
    callback = [TrainingCallback(training_id)
                    ] if training_id != -1 else None
    
    epoch_stage = 0
    if stage_1:
        print("Training network heads")
        epoch_stage = stage_1 + num_stage_1
        model.train(dataset_train, dataset_val,
                    learning_rate = config.LEARNING_RATE,
                    custom_callbacks=callback,
                    epochs = epoch_stage,
                    layers = 'heads',
                    augment = True,
                    augmentation = None)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    if stage_2:
        print("Fine tune Resnet stage 4 and up")
        epoch_stage = epoch_stage + stage_2 
        model.train(dataset_train, dataset_val,
                    learning_rate = config.LEARNING_RATE,
                    custom_callbacks=callback,
                    epochs = epoch_stage,
                    layers = '4+',
                    augment = True,
                    augmentation = None)

    # Training - Stage 3
    # Fine tune all layers
    if stage_3:
        print("Fine tune all layers")
        epoch_stage = epoch_stage + stage_3
        model.train(dataset_train, dataset_val,
                    learning_rate = config.LEARNING_RATE / 10,
                    custom_callbacks=callback,
                    epochs = epoch_stage,
                    layers = 'all',
                    augment = True,
                    augmentation = None)
        
    
# haha
if __name__ == "__main__":
    # from params import DATA_DIR ,LOG_DIR , RPN_ANCHOR_SCALES , BACKBONE, PRETRAIN_MODEL, MODEL_NAME, MODEL_SIZE , STAGE_1 , STAGE_2 , STAGE_3 , TRAIN_STEP , VAL_STEP , ROI_POSITIVE_RATIO , RPN_NMS_TH , MAX_GT_INSTANCES 
    training_model(
        data_dir=DATA_DIR, 
        log_dir=LOG_DIR, 
        rpn_anchor_scales = RPN_ANCHOR_SCALES, 
        backbone = BACKBONE,
        pretrain_model_path=PRETRAIN_MODEL, 
        model_name = MODEL_NAME, 
        model_size = MODEL_SIZE,  
        stage_1=STAGE_1, 
        stage_2=STAGE_2, 
        stage_3=STAGE_3, 
        train_step=TRAIN_STEP, 
        val_step=VAL_STEP, 
        roi_positive_ratio=ROI_POSITIVE_RATIO, 
        rpn_nms_th=RPN_NMS_TH, 
        max_gt_instances=MAX_GT_INSTANCES,
        training_id = TRAINING_ID
    )