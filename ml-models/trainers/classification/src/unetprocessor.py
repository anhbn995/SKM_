from enum import Enum

from model import ArchitechUnet
from utils.unet_data_preparation import *


class UnetType(Enum):
    UNET_50 = 'unet50'
    UNET_BASIC = 'unet_basic'
    UNET_FARM = 'unet_farm'


class UnetRunning(ArchitechUnet):

    def __init__(self, unet_type, **kwargs):
        self.unet_type = unet_type
        super().__init__(**kwargs)

    def identify_model(self):
        if self.unet_type == UnetType.UNET_50.value:
            model = self.get_unet50()
            return model
        elif self.unet_type == UnetType.UNET_FARM.value:
            model = self.get_unetfarm()
            return model

        model = self.get_unet()
        return model

    def preprocess_run(self):
        pass

    def training(self):
        pass

    def predict(self):
        pass

    def postprocess_run(self):
        pass
