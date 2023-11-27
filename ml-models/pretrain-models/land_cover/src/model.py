#Packet định nghĩa các kiến trúc mô hình
#Mô hình phân lớp nhị phân

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, UpSampling2D, MaxPooling2D, Concatenate, Dropout, Dense
from tensorflow.keras.layers import concatenate as merge_l
from tensorflow.keras import optimizers
from unet_utils import *

def unet_basic(input_shape, num_class=1):
    conv_params = dict(activation='relu', padding='same')
    merge_params = dict(axis=-1)
    inputs1 = Input(input_shape)
    conv1 = Convolution2D(32, (3,3), **conv_params)(inputs1)
    conv1 = Convolution2D(32, (3,3), **conv_params)(conv1)
    conv1 = Dropout(rate=0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, (3,3), **conv_params)(pool1)
    conv2 = Convolution2D(64, (3,3), **conv_params)(conv2)
    conv2 = Dropout(rate=0.3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, (3,3), **conv_params)(pool2)
    conv3 = Convolution2D(128, (3,3), **conv_params)(conv3)
    conv3 = Dropout(rate=0.5)(conv3)
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
    conv9 = Dense(3, kernel_regularizer='l1_l2')(conv9)
    # conv9 = Dense(10, kernel_regularizer='l1_l2')(conv9)

    # if have more 1 class (not include background class) 
    # activation func change to 'softmax' and loss to 'categorical_crossentropy'
    conv10 = Convolution2D(num_class, (1, 1), activation='sigmoid')(conv9)
    optimizer = optimizers.Adam(clipvalue=0.5)
    model = Model(inputs=[inputs1], outputs=[conv10])
    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model