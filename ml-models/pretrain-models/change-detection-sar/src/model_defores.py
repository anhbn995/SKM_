import numpy as np
from keras.models import Model
from keras.layers import concatenate as merge_l
from keras.layers import (
    Input, Convolution2D, MaxPooling2D, UpSampling2D,
    Reshape, core, Dropout, Flatten,
    Activation, BatchNormalization, Lambda, Dense, Conv2D, Conv2DTranspose, concatenate,Permute,Cropping2D,Add)
from keras.optimizers import Adam, Nadam, Adadelta,SGD
from keras.losses import binary_crossentropy
from keras import backend as K


def my_acc(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)))

def jaccard_coef(y_true, y_pred):
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def jaccard_coef_int(y_true, y_pred):
    smooth = 1e-12
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def unet_basic(num_channel,size):
    conv_params = dict(activation='relu', padding='same')
    merge_params = dict(axis=-1)
    inputs1 = Input((size, size,int(num_channel)))
    # inputs2 = Input((size, size,int(num_channel)))
    # merge_input = concatenate([inputs1, inputs2])
    conv1 = Convolution2D(32, (3,3), **conv_params)(inputs1)
    conv1 = Convolution2D(32, (3,3), **conv_params)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, (3,3), **conv_params)(pool1)
    conv2 = Convolution2D(64, (3,3), **conv_params)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, (3,3), **conv_params)(pool2)
    conv3 = Convolution2D(128, (3,3), **conv_params)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, (3,3), **conv_params)(pool3)
    conv4 = Convolution2D(256, (3,3), **conv_params)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, (3,3), **conv_params)(pool4)
    conv5 = Convolution2D(512, (3,3), **conv_params)(conv5)

    up6 = merge_l([UpSampling2D(size=(2, 2))(conv5), conv4], **merge_params)
    conv6 = Convolution2D(256, (3,3), **conv_params)(up6)
    conv6 = Convolution2D(256, (3,3), **conv_params)(conv6)

    up7 = merge_l([UpSampling2D(size=(2, 2))(conv6), conv3], **merge_params)
    conv7 = Convolution2D(128, (3,3), **conv_params)(up7)
    conv7 = Convolution2D(128, (3,3), **conv_params)(conv7)

    up8 = merge_l([UpSampling2D(size=(2, 2))(conv7), conv2], **merge_params)
    conv8 = Convolution2D(64, (3,3), **conv_params)(up8)
    conv8 = Convolution2D(64, (3,3), **conv_params)(conv8)

    up9 = merge_l([UpSampling2D(size=(2, 2))(conv8), conv1], **merge_params)
    conv9 = Convolution2D(32, (3,3), **conv_params)(up9)
    conv9 = Convolution2D(32, (3,3), **conv_params)(conv9)

    conv10 = Convolution2D(1, (1, 1), activation='sigmoid')(conv9)
    optimizer=Adam(lr=1e-3)
    # optimizer=SGD(lr=1e-3, decay=1e-8, momentum=0.9, nesterov=True)
    model = Model(input=inputs1, output=conv10)
    model.compile(optimizer=optimizer,
                loss=binary_crossentropy,
                metrics=['accuracy', jaccard_coef, jaccard_coef_int])
    return model


if __name__=="__main__":
    a = unet_basic(3,512)
    # with open('./unet_4band_seperated_dtm.json','w') as f:
    #     model_json = a.to_json()
    #     f.write(model_json)
    a.summary()        
    