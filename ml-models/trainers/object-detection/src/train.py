from config import Config
import keras
import os
import sys
from mrcnn_processor import MrcnnRunning
from dataset import MappingChallengeDataset
import requests
from params import HOSTED_ENDPOINT
ROOT_DIR = os.getcwd()
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


class TrainingCallback(keras.callbacks.Callback):

    def __init__(self, task_id):
        super(TrainingCallback, self).__init__()
        self.task_id = task_id

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        payload = []
        for key in keys:
            payload.append({
                'key': key,
                'value': str(logs[key])
            })
        payload = {
            'task_id': self.task_id,
            'epoch': epoch + 1,
            'payload': payload
        }
        url = '{}/internal/training/report'.format(HOSTED_ENDPOINT)
        print(url)
        print(payload)
        request = requests.post(url, json=payload)
        print(request)


class TrainningConfig(Config):
    def __init__(self, **kwargs):

        # Give the configuration a recognizable name
        # self.NAME = kwargs.get("NAME") or "Model Trainer"

        # We use a GPU with 12GB memory, which can fit two images.
        # Adjust down if you use a smaller GPU.
        self.MEAN_PIXEL = kwargs.get('MEAN_PIXEL')
        self.IMAGES_PER_GPU = kwargs.get("IMAGES_PER_GPU") or 1
        self.IMAGE_RESIZE_MODE = kwargs.get("IMAGE_RESIZE_MODE") or "square"
        self.IMAGE_CHANNEL_COUNT = kwargs.get("IMAGE_CHANNEL_COUNT") or 3

        self.trainer_size = kwargs.get('trainer_size')
        self.IMAGE_MAX_DIM = self.trainer_size+64
        self.IMAGE_MIN_DIM = self.trainer_size+64

        # Number of classes (including background)

        # self.NUM_CLASSES = kwargs.get("NUM_CLASSES") or (1 + 1)  # 1 Backgroun + 1 Building
        super().__init__()

        self.STEPS_PER_EPOCH = kwargs.get("STEPS_PER_EPOCH") or 250
        self.VALIDATION_STEPS = kwargs.get("VALIDATION_STEPS") or 50
        self.RPN_ANCHOR_SCALES = kwargs.get(
            "RPN_ANCHOR_SCALES") or (8, 16, 32, 64, 128)
        self.MAX_GT_INSTANCES = kwargs.get("MAX_GT_INSTANCES") or 300
        self.ROI_POSITIVE_RATIO = kwargs.get("ROI_POSITIVE_RATIO") or 0.66
        self.RPN_NMS_THRESHOLD = kwargs.get("RPN_NMS_THRESHOLD") or 0.7


class MrcnnTrainer(MrcnnRunning):

    def __init__(self, mode, model_dir, data_dir, **kwargs):
        self.config = TrainningConfig(**kwargs)
        super().__init__(mode, self.config, model_dir)
        self.task_id = kwargs.get('task_id')
        self.data_dir = data_dir

        self.epoch1 = kwargs.get('epoch1') or 50
        if kwargs.get('epoch2'):
            self.epoch2 = kwargs.get('epoch2') + self.epoch1
        else:
            self.epoch2 = 100
        if kwargs.get('epoch3'):
            self.epoch3 = kwargs.get('epoch3') + self.epoch2
        else:
            self.epoch3 = 150

    def run(self):
        self.config.display()
        self.model = self.identify_model()
        # Load pretrained weights

        model_path = '{}/model.h5'.format(self.model_dir)
        if os.path.exists(model_path):
            self.load_weights(model_path, by_name=True)
        # on_processing(0.15)
        # Load training dataset
        dataset_train = MappingChallengeDataset()
        dataset_train.load_dataset(dataset_dir=os.path.join(
            self.data_dir, "train"), load_small=False)
        dataset_train.prepare()
        # # Load validation dataset
        dataset_val = MappingChallengeDataset()
        dataset_val.load_dataset(dataset_dir=os.path.join(
            self.data_dir, "val"), load_small=False, return_coco=True)
        dataset_val.prepare()
        print("Training network heads")

        callback = [TrainingCallback(self.task_id)
                    ] if self.task_id != -1 else None

        # on_processing(0.25)
        self.train(dataset_train, dataset_val,
                   learning_rate=self.config.LEARNING_RATE,
                   custom_callbacks=callback,
                   epochs=self.epoch1,
                   # epochs=50,
                   layers='heads', augment=True, augmentation=None)
        # on_processing(0.4)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        self.train(dataset_train, dataset_val,
                   learning_rate=self.config.LEARNING_RATE,
                   epochs=self.epoch2,
                   # epochs=80,
                   custom_callbacks=callback,
                   layers='4+', augment=True, augmentation=None)
        # on_processing(0.65)

        # Training - Stage 3# Fine tune all layers
        print("Fine tune all layers")
        self.train(dataset_train, dataset_val,
                   learning_rate=self.config.LEARNING_RATE / 10,
                   epochs=self.epoch3,
                   # epochs=300,
                   custom_callbacks=callback,
                   layers='all', augment=True, augmentation=None)
        # on_processing(0.9)
        # from numba import cuda
        # cuda.select_device(0)
        # cuda.close()
        return {
            'STEPS_PER_EPOCH': self.config.STEPS_PER_EPOCH,
            'VALIDATION_STEPS': self.config.VALIDATION_STEPS,
            'ROI_POSITIVE_RATIO': self.config.ROI_POSITIVE_RATIO,
            'RPN_NMS_THRESHOLD': self.config.RPN_NMS_THRESHOLD,
            'trainer_size': self.config.trainer_size,
            'numbands': self.config.IMAGE_CHANNEL_COUNT
        }
