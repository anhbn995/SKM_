import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from mrcnn.dataset import TreeCountingDataset
from mrcnn.config import Config
# import imgaug as ia
# from imgaug import augmenters as iaa
import argparse


# def create_seq_augment():
#     """ Define a Sequential augmenters contains some action use for augment use imgaug lib
#     Returns:
#         Sequential augmenters object push to training for augmentation
#     """
#     ia.seed(1)
#     # Example batch of images.
#     # The array has shape (32, 64, 64, 3) and dtype uint8.
#     seq = iaa.Sometimes(0.5, iaa.Sequential([
#         iaa.Fliplr(0.5),
#         # Flip/mirror input images horizontally# horizontal flips
#         iaa.Flipud(0.5),
#         # Flip/mirror input images vertically.
#         iaa.Multiply((0.8, 1.2), per_channel=0.5),
#         # Multiply all pixels in an image with a specific value, thereby making the image darker or brighter.
#         # Multiply 50% of all images with a random value between 0.5 and 1.5
#         # and multiply the remaining 50% channel-wise, i.e. sample one multiplier independently per channel
#         iaa.Affine(
#             scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#             # Scale images to a value of 80 to 120%
#             # of their original size, but do this independently per axis (i.e. sample two values per image)
#             translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
#             # Translate images by -10 to +10% on x- and y-axis independently
#             rotate=(-90, 90),
#             # Rotate images by -90 to 90 degrees
#             #             shear=(-15, 15),
#             #             cval=(0, 255),
#             #             mode=ia.ALL
#         )
#     ]))
#     return seq


def main(data_dir, log_dir, pretrain_model_path=None, model_name="tree_mask_rcnn_tmp", model_size=512, stage_1=None,
         stage_2=None, stage_3=None, train_step=500, val_step=100, roi_positive_ratio=0.33, rpn_nms_th=0.7,
         max_gt_instances=20, augmentation="no"):
    from mrcnn import model as modellib
    model_size = model_size

    class TreeCountingConfig(Config):
        """Configuration for training on data in MS COCO format.
        Derives from the base Config class and overrides values specific
        to the COCO dataset.
        """
        # Give the configuration a recognizable name
        NAME = model_name

        # We use a GPU with 12GB memory, which can fit two images.
        # Adjust down if you use a smaller GPU.
        IMAGES_PER_GPU = 1

        # Uncomment to train on 8 GPUs (default is 1)
        GPU_COUNT = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 1  # 1 Backgroun + 1 Building

        STEPS_PER_EPOCH = train_step
        VALIDATION_STEPS = val_step
        # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
        BACKBONE = "resnet101"
        IMAGE_RESIZE_MODE = "square"
        DETECTION_MAX_INSTANCES = 100
        MAX_GT_INSTANCES = max_gt_instances
        RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
        ROI_POSITIVE_RATIO = roi_positive_ratio
        RPN_NMS_THRESHOLD = rpn_nms_th
        # RPN_NMS_THRESHOLD = 0.9
        # ROI_POSITIVE_RATIO = 0.5
        IMAGE_MAX_DIM = model_size + 64
        IMAGE_MIN_DIM = model_size + 64

    config = TreeCountingConfig()
    config.display()
    if augmentation == "yes":
        print(augmentation)
        # seq = create_seq_augment()
    else:
        seq = None
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=log_dir)
    # Load pretrained weights
    if pretrain_model_path:
        model_path = pretrain_model_path
        model.load_weights(model_path, by_name=True)
    # Load training dataset
    dataset_train = TreeCountingDataset()
    dataset_train.load_dataset(dataset_dir=os.path.join(data_dir, "train"), load_small=False)
    dataset_train.prepare()

    # # Load validation dataset
    dataset_val = TreeCountingDataset()
    val_coco = dataset_val.load_dataset(dataset_dir=os.path.join(data_dir, "val"), load_small=False, return_coco=True)
    dataset_val.prepare()
    if stage_1:
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=stage_1,
                    layers='heads',
                    augment=True,
                    augmentation=None)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    if stage_2:
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=stage_2,
                    layers='4+',
                    augment=True,
                    augmentation=None)

    # Training - Stage 3
    # Fine tune all layers

    if stage_3:
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=stage_3,
                    layers='all',
                    augment=True,
                    augmentation=None)


# haha
if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        '--pre_train',
        help='Pre train model path for transfer learning',
    )

    args_parser.add_argument(
        '--data_dir',
        help='Train data directory',
        required=True
    )

    args_parser.add_argument(
        '--log_dir',
        help='Log directory for store model weight and log',
        required=True
    )
    args_parser.add_argument(
        '--model_name',
        default="tree_mask_rcnn",
        help='Name of model',
    )

    args_parser.add_argument(
        '--model_size',
        help='Size of model',
        default=512,
        type=int,
    )
    args_parser.add_argument(
        '--stage_1',
        help='Num epoch for training network heads',
        type=int,
    )

    args_parser.add_argument(
        '--stage_2',
        help='Num epoch for fine tune layer Resnet stage 4 and up',
        type=int,
    )

    args_parser.add_argument(
        '--stage_3',
        help='Num epoch for fine tune all network layers',
        type=int,
    )
    args_parser.add_argument(
        '--rps_ratio',
        help='Roi positive ratio',
        type=float,
        default=0.33
    )
    args_parser.add_argument(
        '--rpn_ratio',
        help='Rpn threshold ratio',
        type=float,
        default=0.7
    )
    args_parser.add_argument(
        '--max_gt',
        help='Max number ground truth instance each image',
        type=int,
        default=20
    )
    args_parser.add_argument(
        '--train_step',
        help='Number train step for each epoch while training',
        type=int,
        default=100
    )
    args_parser.add_argument(
        '--val_step',
        help='Number validate step for each epoch while training',
        type=int,
        default=100
    )
    args_parser.add_argument(
        '--augment',
        help='Choices Augmentation while training or not, must be "yes" or "no"',
        type=str,
        choices=["yes", "no"],
        default="no"
    )
    param = args_parser.parse_args()
    data_dir = param.data_dir
    log_dir = param.log_dir
    pre_train = param.pre_train
    model_name = param.model_name
    model_size = param.model_size
    stage_1 = param.stage_1
    stage_2 = param.stage_2
    stage_3 = param.stage_3
    rps_ratio = param.rps_ratio
    rpn_ratio = param.rpn_ratio
    max_gt = param.max_gt
    train_step = param.train_step
    val_step = param.val_step
    augment = param.augment
    main(data_dir, log_dir, pretrain_model_path=pre_train, model_name=model_name,
         model_size=model_size, stage_1=stage_1, stage_2=stage_2, stage_3=stage_3,
         roi_positive_ratio=rps_ratio, rpn_nms_th=rpn_ratio, max_gt_instances=max_gt,
         train_step=train_step, val_step=val_step, augmentation=augment)


"""
python train_tree_counting.py --data_dir /home/skm/SKM16/Tmp/TreeCounting/Test_EOF/TrainingDataSet/traning_fasterRcnn --log_dir /home/skm/SKM16/Tmp/TreeCounting/Test_EOF/Model --stage_1 20 --stage_2 10 --stage_3 5
"""