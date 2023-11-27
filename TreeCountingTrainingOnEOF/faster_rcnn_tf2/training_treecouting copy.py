import os, re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import argparse
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn.dataset import TreeCountingDataset


def training_model(data_dir, log_dir, rpn_anchor_scales, backbone,
                   pretrain_model_path=None, model_name="tree_mask_rcnn", model_size=256,  
                   stage_1=None, stage_2=None, stage_3=None, train_step=500, val_step=100, 
                   roi_positive_ratio=0.33, rpn_nms_th=0.7, max_gt_instances=100):
    
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
    
    
    epoch_stage = 0
    if stage_1:
        print("Training network heads")
        epoch_stage = stage_1 + num_stage_1
        model.train(dataset_train, dataset_val,
                    learning_rate = config.LEARNING_RATE,
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
                    epochs = epoch_stage,
                    layers = 'all',
                    augment = True,
                    augmentation = None)
        
    
# haha
if __name__ == "__main__":
    import time
    x = time.time()
    data_dir=r'/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/data_faster-rnn/image_crop_box/tmp/gen_TauBien_Original_3band_cut_128_stride_64_time_20230710_011308/images_data'
    log_dir=r'/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/data_faster-rnn/image_crop_box/tmp/gen_TauBien_Original_3band_cut_128_stride_64_time_20230710_011308/images_data/Model'
    rpn_anchor_scales =(8, 16, 32, 64, 128)
    backbone = "resnet50"
    # pretrain_model_path=None, 
    model_name="tauBien"
    model_size=128
    stage_1=40
    stage_2=30
    stage_3=80
    train_step=500
    val_step=200
    roi_positive_ratio=0.33
    rpn_nms_th=0.3
    max_gt_instances=50

    training_model(data_dir=data_dir, 
                    log_dir=log_dir, 
                    rpn_anchor_scales=rpn_anchor_scales,
                    backbone=backbone,
                    # pretrain_model_path=None, 
                    model_name=model_name,
                    model_size=model_size,  
                    stage_1=stage_1, stage_2=stage_2, stage_3=stage_3, train_step=train_step, val_step=val_step, 
                    roi_positive_ratio=roi_positive_ratio, rpn_nms_th=rpn_nms_th, max_gt_instances=max_gt_instances)
    y = time.time() - x
    print(y)
    file_path = os.path.join(log_dir, "model_config.txt")
    my_list = ["data_dir", "log_dir", "rpn_anchor_scales", "backbone", "model_name","model_size","stage_1","stage_2","stage_3","train_step","val_step","roi_positive_ratio","rpn_nms_th","max_gt_instances", "y"]
    value_ = [data_dir, log_dir, str(rpn_anchor_scales), str(backbone), str(model_name), str(model_size), str(stage_1), str(stage_2), str(stage_3), str(train_step), str(val_step), str(roi_positive_ratio), str(rpn_nms_th), str(max_gt_instances), str(y)]
    with open(file_path, "w") as file:
    # Write each item from the list to a new line in the file
        for idx, item in enumerate(my_list):
            file.write(item + ": " + value_[idx] + "\n")
        
        
# 256 11647.111773729324
# 128 11394.186260461807