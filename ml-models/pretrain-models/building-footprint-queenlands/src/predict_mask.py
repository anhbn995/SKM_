from tensorflow.keras import layers, Model, regularizers, Input, backend
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
import os
import numpy as np
import glob
import cv2
# import matplotlib.pyplot as plt
import rasterio
import shutil
import geopandas as gp
import pandas as pd
import cv2
from rasterio.windows import Window
import os
from shapely.geometry import box as bbbox
from shapely.geometry import Polygon
import random
import shapely
import tqdm
from PIL import Image

class DataParser():
    def __init__(self, annotation):
        self.total_data = annotation
        self.batch_size = 8
        self.steps_per_epoch = int(len(self.total_data)//self.batch_size)
        self.check_batch = self.steps_per_epoch * self.batch_size
        self.augmentations = [self.flip_ud, self.flip_lr, self.rot90]
        self.num = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.num < self.check_batch:
            filename = self.total_data[self.num: self.num+self.batch_size]

            image, label = self.get_batch(filename)
            self.num += self.batch_size
            return image, label
        else:
            self.num = 0
            np.random.shuffle(self.total_data)
            raise StopIteration


    def get_batch(self, batch):
        images = []
        edgemaps = []
        for img_list in batch:
            im, em = self.preprocess(img_list, img_list.replace('image', 'mask'))

            for f in self.augmentations:
                if np.random.uniform()<0.20:
                    im, em=f(im, em)

            images.append(im)
            edgemaps.append(em)

        images   = np.asarray(images)
        edgemaps = np.asarray(edgemaps)

        return images, edgemaps

    def preprocess(self, path_img, path_mask):
        with rasterio.open(path_img) as img:
            width,height = img.width,img.height
            new_image_width = new_image_height = max(width,height)
            values = img.read().transpose(1,2,0).astype(np.uint8)
            x_center = (new_image_width - width) // 2
            y_center = (new_image_height - height) // 2
            result = np.full((new_image_height,new_image_width, 3), (0,0,0), dtype=np.uint8)
            result[y_center:y_center+height, x_center:x_center+width] = values
            result = cv2.resize(result,(320,320), interpolation = cv2.INTER_CUBIC)
            image = result/255.0
        with rasterio.open(path_mask) as mas:
            values = mas.read()
            result = np.full((new_image_height,new_image_width), 0, dtype=np.uint8)
            result[y_center:y_center+height, x_center:x_center+width] = values[0]
            result = cv2.resize(result,(320,320), interpolation = cv2.INTER_CUBIC)
            label = (result/255.0 > 0.5).astype(np.float32)
        return image, label[...,np.newaxis]
    
    def flip_ud(self, im, em):
        return np.flipud(im), np.flipud(em)

    def flip_lr(self, im, em):
        return np.fliplr(im), np.fliplr(em)

    def rot90(self, im, em):
        return np.rot90(im), np.rot90(em)
    
    def __len__(self):
        return self.steps_per_epoch
    

l2 = regularizers.l2
w_decay=1e-3
weight_init = tf.initializers.glorot_uniform()

def DoubleConvBlock(input, mid_features, out_features=None, stride=(1,1), use_bn=True,use_act=True):
    out_features = mid_features if out_features is None else out_features
    k_reg = None if w_decay is None else l2(w_decay)
    x = layers.Conv2D(filters=mid_features, kernel_size=(3, 3), strides=stride, padding='same', kernel_initializer=weight_init, kernel_regularizer=k_reg)(input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=out_features, kernel_size=(3, 3), strides=(1,1), padding='same', kernel_initializer=weight_init, kernel_regularizer=k_reg)(x)
    x = layers.BatchNormalization()(x)
    if use_act:
        x = layers.ReLU()(x)
    return x

def SingleConvBlock(input, out_features, k_size=(1,1),stride=(1,1), use_bs=False, use_act=False, w_init=None):
    k_reg = None if w_decay is None else l2(w_decay)
    x = layers.Conv2D(filters=out_features, kernel_size=k_size, strides=stride, padding='same',kernel_initializer=w_init, kernel_regularizer=k_reg)(input)
    if use_bs:
        x = layers.BatchNormalization()(x)
    if use_act:
        x = layers.ReLU()(x)
    return x

def UpConvBlock(input_data, up_scale):
    total_up_scale = 2 ** up_scale
    constant_features = 16
    k_reg = None if w_decay is None else l2(w_decay)
    features = []
    for i in range(up_scale):
        out_features = 1 if i == up_scale-1 else constant_features
        if i==up_scale-1:
            input_data = layers.Conv2D(filters=out_features, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu', kernel_initializer=tf.initializers.TruncatedNormal(mean=0.), kernel_regularizer=k_reg,use_bias=True)(input_data)
            input_data = layers.Conv2DTranspose(out_features, kernel_size=(total_up_scale,total_up_scale), strides=(2,2), padding='same', kernel_initializer=tf.initializers.TruncatedNormal(stddev=0.1), kernel_regularizer=k_reg,use_bias=True)(input_data)
        else:
            input_data = layers.Conv2D(filters=out_features, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu',kernel_initializer=weight_init, kernel_regularizer=k_reg,use_bias=True)(input_data)
            input_data = layers.Conv2DTranspose(out_features, kernel_size=(total_up_scale,total_up_scale),strides=(2,2), padding='same', use_bias=True, kernel_initializer=weight_init, kernel_regularizer=k_reg)(input_data)
    return input_data

def _DenseLayer(inputs, out_features):
    k_reg = None if w_decay is None else l2(w_decay)
    x, x2 = tuple(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=out_features, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer=weight_init, kernel_regularizer=k_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=out_features, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer=weight_init, kernel_regularizer=k_reg)(x)
    x = layers.BatchNormalization()(x)
    return 0.5 * (x + x2), x2

def _DenseBlock(input_da, num_layers, out_features):
    for i in range(num_layers):
        input_da = _DenseLayer(input_da, out_features)
    return input_da

def DexiNed(image_band_channel):
    img_input = Input(shape=(480,480,image_band_channel), name='input')

    block_1 = DoubleConvBlock(img_input, 32, 64, stride=(2,2),use_act=False)
    block_1_side = SingleConvBlock(block_1, 128, k_size=(1,1),stride=(2,2),use_bs=True, w_init=weight_init)

    # Block 2
    block_2 = DoubleConvBlock(block_1, 128, use_act=False)
    block_2_down = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(block_2)
    block_2_add = block_2_down + block_1_side
    block_2_side = SingleConvBlock(block_2_add, 256,k_size=(1,1),stride=(2,2),use_bs=True, w_init=weight_init)

    # Block 3
    block_3_pre_dense = SingleConvBlock(block_2_down,256,k_size=(1,1),stride=(1,1),use_bs=True,w_init=weight_init)
    block_3, _ = _DenseBlock([block_2_add, block_3_pre_dense], 2, 256)
    block_3_down = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(block_3)
    block_3_add = block_3_down + block_2_side
    block_3_side = SingleConvBlock(block_3_add, 512,k_size=(1,1),stride=(2,2),use_bs=True,w_init=weight_init)

    # Block 4
    block_4_pre_dense_256 = SingleConvBlock(block_2_down, 256,k_size=(1,1),stride=(2,2), w_init=weight_init)
    block_4_pre_dense = SingleConvBlock(block_4_pre_dense_256 + block_3_down, 512,k_size=(1,1),stride=(1,1),use_bs=True, w_init=weight_init)
    block_4, _ = _DenseBlock([block_3_add, block_4_pre_dense], 3, 512)
    block_4_down = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(block_4)
    block_4_add = block_4_down + block_3_side
    block_4_side = SingleConvBlock(block_4_add, 512,k_size=(1,1),stride=(1,1),use_bs=True, w_init=weight_init)

    # Block 5
    block_5_pre_dense_512 = SingleConvBlock(block_4_pre_dense_256, 512, k_size=(1,1),stride=(2,2), w_init=weight_init)
    block_5_pre_dense = SingleConvBlock(block_5_pre_dense_512 + block_4_down, 512,k_size=(1,1),stride=(1,1),use_bs=True, w_init=weight_init)
    block_5, _ = _DenseBlock([block_4_add, block_5_pre_dense], 3, 512)
    block_5_add = block_5 + block_4_side

    # Block 6
    block_6_pre_dense = SingleConvBlock(block_5, 256,k_size=(1,1),stride=(1,1),use_bs=True, w_init=weight_init)
    block_6, _ =  _DenseBlock([block_5_add, block_6_pre_dense], 3, 256)


    out_1 = UpConvBlock(block_1, 1)
    out_2 = UpConvBlock(block_2, 1)
    out_3 = UpConvBlock(block_3, 2)
    out_4 = UpConvBlock(block_4, 3)
    out_5 = UpConvBlock(block_5, 4)
    out_6 = UpConvBlock(block_6, 4)

    # concatenate multiscale outputs
    block_cat = tf.concat([out_1, out_2, out_3, out_4, out_5, out_6], 3)  # BxHxWX6
    block_cat = SingleConvBlock(block_cat, 1,k_size=(1,1),stride=(1,1), w_init=tf.constant_initializer(1/5))  # BxHxWX1
    
    block_cat = layers.Activation('sigmoid')(block_cat)
    out_1 = layers.Activation('sigmoid')(out_1)
    out_2 = layers.Activation('sigmoid')(out_2)
    out_3 = layers.Activation('sigmoid')(out_3)
    out_4 = layers.Activation('sigmoid')(out_4)
    out_5 = layers.Activation('sigmoid')(out_5)
    out_6 = layers.Activation('sigmoid')(out_6)

    model = Model(inputs=[img_input], outputs=[block_cat, out_1, out_2, out_3, out_4, out_5, out_6])
    # model.summary()

    return model

def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    # x = layers.BatchNormalization(axis=bn_axis, scale=False, center=False,name=bn_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x

def InceptionV3(img_input):
    output = []
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = layers.ZeroPadding2D(35)(img_input)
    x = conv2d_bn(x, 32, 3, 3, strides=(1, 1), padding='valid')
    x = conv2d_bn(x, 32, 3, 3)
    x = conv2d_bn(x, 64, 3, 3)
    output.append(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3)
    output.append(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1, padding='valid')

    branch5x5 = conv2d_bn(x, 48, 1, 1, padding='valid')
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, padding='valid')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1, padding='valid')
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1, padding='valid')

    branch5x5 = conv2d_bn(x, 48, 1, 1, padding='valid')
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, padding='valid')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, padding='valid')
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1, padding='valid')

    branch5x5 = conv2d_bn(x, 48, 1, 1, padding='valid')
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, padding='valid')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, padding='valid')
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed2')
    output.append(x)
    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding = 'valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, padding= 'valid')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding = 'valid')

    # branch_pool = layers.ZeroPadding2D(1)(x)
    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1, padding = 'valid')

    branch7x7 = conv2d_bn(x, 128, 1, 1, padding = 'valid')
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1, padding = 'valid')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, padding = 'valid')
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1, padding = 'valid')

        branch7x7 = conv2d_bn(x, 160, 1, 1, padding = 'valid')
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1, padding = 'valid')
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D( (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1, padding = 'valid')
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1, padding = 'valid')

    branch7x7 = conv2d_bn(x, 192, 1, 1, padding = 'valid')
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1, padding = 'valid')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, padding = 'valid')
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name='mixed7')
    output.append(x)

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1, padding = 'valid')
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1, padding = 'valid')
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1, padding = 'valid')

        branch3x3 = conv2d_bn(x, 384, 1, 1, padding = 'valid')
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1, padding = 'valid')
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1, padding = 'valid')
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed' + str(9 + i))
    output.append(x)

    return output

def crop(d, region):
    x, y, h, w = region
    d1 = d[:, x:x + h, y:y + w, :]
    return d1

def DilConv(x, kernel_size, padding, dilation, stride = 1, C_out = 64):
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding)(x)
    x = layers.Conv2D(C_out, kernel_size=kernel_size, strides=stride, dilation_rate=dilation)(x)
    return x

def Conv(x, kernel_size, padding, stride = 1, C_out = 64):
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding)(x)
    x = layers.Conv2D(C_out, kernel_size=kernel_size, strides=stride)(x)
    return x

def Identity(x):
    return x

def cell1(x, flag=1):
    x1 = Conv(x, 5, 2)
    out_x = x + x1
    if flag == 1:
        out_x = layers.UpSampling2D(interpolation = 'bilinear')(out_x)
    return out_x

def cell2(x, flag=1):
    x1 = DilConv(x, 3, 2, 2)
    out_x = x + x1
    if flag == 1:
        out_x = layers.UpSampling2D(interpolation = 'bilinear')(out_x)
    return out_x

def cell3(x, flag=1):
    x1 = Conv(x, 3, 1)
    out_x = x + x1
    if flag == 1:
        out_x = layers.UpSampling2D(interpolation = 'bilinear')(out_x)
    return out_x

def cell4(x, flag=1):
    x1 = DilConv(x, 5, 8, 4)
    out_x = x + x1
    if flag == 1:
        out_x = layers.UpSampling2D(interpolation = 'bilinear')(out_x)
    return out_x

def cell_fuse(x, flag=1):
    x1 = DilConv(x, 3, 2, 2)
    out_x = x + x1
    if flag == 1:
        out_x = layers.UpSampling2D(interpolation = 'bilinear')(out_x)
    return out_x

def NetWork(shape, C = 64):
    # input = Input(shape=(299,299,shape))
    input = Input(shape=(320,320,shape))
    size = input.shape[1:3]
    conv1, conv2, conv3, conv4, conv5 = InceptionV3(input)
    dsn1 = layers.Conv2D(C, 1)(conv1)
    dsn2 = layers.Conv2D(C, 1)(conv2)
    dsn3 = layers.Conv2D(C, 1)(conv3)
    dsn4 = layers.Conv2D(C, 1)(conv4)
    dsn5 = layers.Conv2D(C, 1)(conv5)
    c1 = cell1(dsn5)

    mm1 = layers.concatenate([c1, crop(dsn4, (0, 0)+ c1.shape[1:3])])
    mm1 = layers.Conv2D(C, 1)(mm1)
    d4_2 = layers.Activation('relu')(mm1)
    c2 = cell2(d4_2)

    mm2 = layers.UpSampling2D(interpolation= 'bilinear')(c1)
    mm2 = layers.concatenate([mm2, crop(dsn3, (0, 0) + mm2.shape[1:3])])
    mm2 = layers.Conv2D(C, 1)(mm2)
    d3_2 = layers.Activation('relu')(mm2)
    d3_2 = layers.concatenate([c2, crop(d3_2, (0, 0) + c2.shape[1:3])])
    d3_2 = layers.Conv2D(C, 1)(d3_2)
    d3_2 = layers.Activation('relu')(d3_2)
    c3 = cell3(d3_2)

    c4 = cell4(dsn2)

    d_fuse = tf.zeros_like(c3)
    d_fuse = crop(layers.UpSampling2D(interpolation="bilinear")(c2), (0,0) + d_fuse.shape[1:3]) + crop(c3, (0, 0) + c3.shape[1:3]) + crop(layers.MaxPool2D()(c4), (0, 0) + d_fuse.shape[1:3])
    d_fuse = cell_fuse(d_fuse)
    d_fuse = layers.Conv2D(1, 1)(d_fuse)
    d_fuse = layers.ZeroPadding2D(7)(d_fuse)
    sss = layers.Conv2D(1, 15)(d_fuse)

    out_fuse = crop(sss, (34, 34) + size)
    out = crop(layers.Conv2D(1, 1)(layers.UpSampling2D(size=(4, 4), interpolation = 'bilinear')(c2)), (34, 34) + size)

    out_fuse = layers.Activation('sigmoid')(out_fuse)
    out = layers.Activation('sigmoid')(out)
    model = Model(input, [out_fuse, out], name='inception_v3')
    # model.summary()
    return model
    
def _upsample_like(src,tar):
    # src = tf.image.resize(images=src, size=tar.shape[1:3], method= 'bilinear')
    h = int(tar.shape[1]/src.shape[1])
    w = int(tar.shape[2]/src.shape[2])
    src = layers.UpSampling2D((h,w),interpolation='bilinear')(src)
    return src

def REBNCONV(x,out_ch=3,dirate=1):
    # x = layers.ZeroPadding2D(1*dirate)(x)
    x = layers.Conv2D(out_ch, 3, padding = "same", dilation_rate=1*dirate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def RSU7(hx, mid_ch=12, out_ch=3):
    hxin = REBNCONV(hx, out_ch,dirate=1)

    hx1 = REBNCONV(hxin, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx1)

    hx2 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx2)

    hx3 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx3)

    hx4 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx4)

    hx5 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx5)

    hx6 = REBNCONV(hx, mid_ch,dirate=1)

    hx7 = REBNCONV(hx6, mid_ch,dirate=2)

    hx6d = REBNCONV(layers.concatenate([hx7,hx6]), mid_ch,dirate=1)
    hx6dup = _upsample_like(hx6d,hx5)

    hx5d = REBNCONV(layers.concatenate([hx6dup,hx5]), mid_ch,dirate=1)
    hx5dup = _upsample_like(hx5d,hx4)

    hx4d = REBNCONV(layers.concatenate([hx5dup,hx4]), mid_ch,dirate=1)
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = REBNCONV(layers.concatenate([hx4dup,hx3]), mid_ch,dirate=1)
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = REBNCONV(layers.concatenate([hx3dup,hx2]), mid_ch,dirate=1)
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = REBNCONV(layers.concatenate([hx2dup,hx1]), out_ch,dirate=1)

    return hx1d + hxin

def RSU6(hx, mid_ch=12, out_ch=3):
    hxin = REBNCONV(hx, out_ch,dirate=1)
    
    hx1 = REBNCONV(hxin, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx1)

    hx2 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx2)

    hx3 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx3)

    hx4 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx4)

    hx5 = REBNCONV(hx, mid_ch,dirate=1)

    hx6 = REBNCONV(hx, mid_ch,dirate=2)


    hx5d =  REBNCONV(layers.concatenate([hx6,hx5]), mid_ch,dirate=1)
    hx5dup = _upsample_like(hx5d,hx4)

    hx4d = REBNCONV(layers.concatenate([hx5dup,hx4]), mid_ch,dirate=1)
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = REBNCONV(layers.concatenate([hx4dup,hx3]), mid_ch,dirate=1)
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = REBNCONV(layers.concatenate([hx3dup,hx2]), mid_ch,dirate=1)
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = REBNCONV(layers.concatenate([hx2dup,hx1]), out_ch,dirate=1)

    return hx1d + hxin

def RSU5(hx, mid_ch=12, out_ch=3):
    hxin = REBNCONV(hx, out_ch,dirate=1)

    hx1 = REBNCONV(hxin, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx1)

    hx2 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx2)

    hx3 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx3)

    hx4 = REBNCONV(hx, mid_ch,dirate=1)

    hx5 = REBNCONV(hx4, mid_ch,dirate=2)

    hx4d = REBNCONV(layers.concatenate([hx5,hx4]), mid_ch,dirate=1)
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = REBNCONV(layers.concatenate([hx4dup,hx3]), mid_ch,dirate=1)
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = REBNCONV(layers.concatenate([hx3dup,hx2]), mid_ch,dirate=1)
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = REBNCONV(layers.concatenate([hx2dup,hx1]), out_ch,dirate=1)

    return hx1d + hxin


def RSU4(hx,mid_ch=12, out_ch=3):
    hxin = REBNCONV(hx, out_ch,dirate=1)

    hx1 = REBNCONV(hxin,mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx1)

    hx2 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx2)

    hx3 = REBNCONV(hx, mid_ch,dirate=1)

    hx4 = REBNCONV(hx3, mid_ch,dirate=2)
    hx3d = REBNCONV(layers.concatenate([hx4,hx3]), mid_ch,dirate=1)
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = REBNCONV(layers.concatenate([hx3dup,hx2]), mid_ch,dirate=1)
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = REBNCONV(layers.concatenate([hx2dup,hx1]), out_ch,dirate=1)

    return hx1d + hxin

def RSU4F(hx, mid_ch=12, out_ch=3):
    hxin = REBNCONV(hx,out_ch,dirate=1)

    hx1 = REBNCONV(hxin, mid_ch,dirate=1)
    hx2 = REBNCONV(hx1, mid_ch,dirate=2)
    hx3 = REBNCONV(hx2, mid_ch,dirate=4)

    hx4 = REBNCONV(hx3, mid_ch,dirate=8)

    hx3d = REBNCONV(layers.concatenate([hx4,hx3]), mid_ch,dirate=4)
    hx2d = REBNCONV(layers.concatenate([hx3d,hx2]), mid_ch,dirate=2)
    hx1d = REBNCONV(layers.concatenate([hx2d,hx1]), out_ch,dirate=1)

    return hx1d + hxin

def U2NET(hx, out_ch=1):
    # hx = Input(shape=(480,480,3))
    #stage 1
    hx1 = RSU7(hx, 32,64)
    hx = layers.MaxPool2D(2,strides=2)(hx1)

    #stage 2
    hx2 = RSU6(hx, 32,128)
    hx = layers.MaxPool2D(2,strides=2)(hx2)

    #stage 3
    hx3 = RSU5(hx, 64,256)
    hx = layers.MaxPool2D(2,strides=2)(hx3)

    #stage 4
    hx4 = RSU4(hx, 128,512)
    hx = layers.MaxPool2D(2,strides=2)(hx4)

    #stage 5
    hx5 = RSU4F(hx, 256,512)
    hx = layers.MaxPool2D(2,strides=2)(hx5)

    #stage 6
    hx6 = RSU4F(hx, 256,512)
    hx6up = _upsample_like(hx6,hx5)

    #-------------------- decoder --------------------
    hx5d = RSU4F(layers.concatenate([hx6up,hx5]), 256,512)
    hx5dup = _upsample_like(hx5d,hx4)

    hx4d = RSU4(layers.concatenate([hx5dup,hx4]), 128,256)
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = RSU5(layers.concatenate([hx4dup,hx3]), 64,128)
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = RSU6(layers.concatenate([hx3dup,hx2]), 32,64)
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = RSU7(layers.concatenate([hx2dup,hx1]), 16,64)


    #side output
    d1 = layers.Conv2D(1, 3,padding="same")(hx1d)

    d2 = layers.Conv2D(1, 3,padding="same")(hx2d)
    d2 = _upsample_like(d2,d1)

    d3 = layers.Conv2D(1, 3,padding="same")(hx3d)
    d3 = _upsample_like(d3,d1)

    d4 = layers.Conv2D(1, 3,padding="same")(hx4d)
    d4 = _upsample_like(d4,d1)

    d5 = layers.Conv2D(1, 3,padding="same")(hx5d)
    d5 = _upsample_like(d5,d1)

    d6 = layers.Conv2D(1, 3,padding="same")(hx6)
    d6 = _upsample_like(d6,d1)

    d0 = layers.Conv2D(out_ch,1)(layers.concatenate([d1,d2,d3,d4,d5,d6]))

    o1    = layers.Activation('sigmoid')(d1)
    o2    = layers.Activation('sigmoid')(d2)
    o3    = layers.Activation('sigmoid')(d3)
    o4    = layers.Activation('sigmoid')(d4)
    o5    = layers.Activation('sigmoid')(d5)
    o6    = layers.Activation('sigmoid')(d6)
    ofuse = layers.Activation('sigmoid')(d0)

    return [ofuse, o1, o2, o3, o4, o5, o6]

def U2NETP(hx, out_ch=1):
    # hx = Input(shape=(480,480,3))
    #stage 1
    hx1 = RSU7(hx, 16,64)
    hx = layers.MaxPool2D(2,strides=2)(hx1)

    #stage 2
    hx2 = RSU6(hx, 16,64)
    hx = layers.MaxPool2D(2,strides=2)(hx2)

    #stage 3
    hx3 = RSU5(hx, 16,64)
    hx = layers.MaxPool2D(2,strides=2)(hx3)

    #stage 4
    hx4 = RSU4(hx, 16,64)
    hx = layers.MaxPool2D(2,strides=2)(hx4)

    #stage 5
    hx5 = RSU4F(hx, 16,64)
    hx = layers.MaxPool2D(2,strides=2)(hx5)

    #stage 6
    hx6 = RSU4F(hx, 16,64)
    hx6up = _upsample_like(hx6,hx5)

    #-------------------- decoder --------------------
    hx5d = RSU4F(layers.concatenate([hx6up,hx5]), 16,64)
    hx5dup = _upsample_like(hx5d,hx4)

    hx4d = RSU4(layers.concatenate([hx5dup,hx4]), 16,64)
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = RSU5(layers.concatenate([hx4dup,hx3]), 16,64)
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = RSU6(layers.concatenate([hx3dup,hx2]), 16,64)
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = RSU7(layers.concatenate([hx2dup,hx1]), 16,64)


    #side output
    d1 = layers.Conv2D(1, 3,padding="same")(hx1d)

    d2 = layers.Conv2D(1, 3,padding="same")(hx2d)
    d2 = _upsample_like(d2,d1)

    d3 = layers.Conv2D(1, 3,padding="same")(hx3d)
    d3 = _upsample_like(d3,d1)

    d4 = layers.Conv2D(1, 3,padding="same")(hx4d)
    d4 = _upsample_like(d4,d1)

    d5 = layers.Conv2D(1, 3,padding="same")(hx5d)
    d5 = _upsample_like(d5,d1)

    d6 = layers.Conv2D(1, 3,padding="same")(hx6)
    d6 = _upsample_like(d6,d1)

    d0 = layers.Conv2D(out_ch,1)(layers.concatenate([d1,d2,d3,d4,d5,d6]))

    o1    = layers.Activation('sigmoid')(d1)
    o2    = layers.Activation('sigmoid')(d2)
    o3    = layers.Activation('sigmoid')(d3)
    o4    = layers.Activation('sigmoid')(d4)
    o5    = layers.Activation('sigmoid')(d5)
    o6    = layers.Activation('sigmoid')(d6)
    ofuse = layers.Activation('sigmoid')(d0)

    return tf.stack([ofuse, o1, o2, o3, o4, o5, o6])


def Model_U2Net(num_band):
    hx = Input(shape=(320,320,num_band))
    out = U2NET(hx)
    model = Model(inputs = hx, outputs = out)
    return model

def pre_process_binary_cross_entropy(label, inputs):
    # preprocess data
    y = label
    loss = 0
    w_loss=1.0
    for tmp_p in inputs:
        tmp_y = tf.cast(y, dtype=tf.float32)
        mask = tf.dtypes.cast(tmp_y > 0., tf.float32)
        b,h,w,c=mask.get_shape()
        positives = tf.math.reduce_sum(mask, axis=[1, 2, 3], keepdims=True)
        negatives = h*w*c-positives

        beta2 = positives / (negatives + positives) # negatives in hed
        beta = negatives / (positives + negatives) # positives in hed
        pos_w = tf.where(tf.equal(y, 0.0), beta2, beta)
        
        l_cost = bce(y_true=tmp_y, y_pred=tmp_p, sample_weight=pos_w)
        loss += (l_cost*w_loss)
    return loss

def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def weighted_cross_entropy_loss(label, inputs):
    loss = 0
    y = tf.cast(label, dtype=tf.float32)
    negatives = tf.math.reduce_sum(1.-y)
    positives = tf.math.reduce_sum(y)
    beta = negatives/(negatives + positives)
    pos_w = beta/(1-beta)

    for predict in inputs:
        _epsilon = _to_tensor(tf.keras.backend.epsilon(), predict.dtype.base_dtype)
        predict   = tf.clip_by_value(predict, _epsilon, 1 - _epsilon)
        predict   = tf.math.log(predict/(1 - predict))
        
        cost = tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=predict, pos_weight=pos_w)
        cost = tf.math.reduce_mean(cost*(1-beta))
        loss += tf.where(tf.equal(positives, 0.0), 0.0, cost)
    return loss

def train_step(image_data, target):
    with tf.GradientTape() as tape:
        pred = my_model(image_data, training=True)
        loss = weighted_cross_entropy_loss(target, pred)

        gradients = tape.gradient(loss, my_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, my_model.trainable_variables))

        global_steps.assign_add(1)

        with writer.as_default():
            tf.summary.scalar("loss/loss", loss, step=global_steps)
        writer.flush()     
    return global_steps.numpy(), loss.numpy()

def validate_step(image_data, target):
    pred = my_model(image_data, training=False)
    loss = weighted_cross_entropy_loss(target, pred)
    return loss.numpy()

def transform_geom_to_poly_px(polygon, geotransform):
    geo_polygon_bound = np.dstack(polygon.exterior.coords.xy)[0]
    top_left_x = geotransform[2]
    top_left_y = geotransform[5]
    x_res = geotransform[0]
    y_res = geotransform[4]
    poly = np.array(geo_polygon_bound)
    poly_px = (poly - np.array([top_left_x, top_left_y])) / np.array([x_res, y_res])
    return poly_px.round()

def transform_poly_px_to_geom(polygon, geotransform):
    top_left_x = geotransform[2]
    top_left_y = geotransform[5]
    x_res = geotransform[0]
    y_res = geotransform[4]
    poly = np.array(polygon)
    poly_rs = poly * np.array([x_res, y_res]) + np.array([top_left_x, top_left_y])
    return poly_rs
def extend_bound(bound_geom):
    x_scale = 1 + float(random.randint(60, 80))/200
    y_scale = 1 + float(random.randint(60, 80))/200
    bound_geom = shapely.affinity.scale(bound_geom, xfact=1.3, yfact=1.3, origin='center')
    return bound_geom

def predict_building(image_path,shape_path,out_path,model_path):
    INPUT_SIZE = 320
    model = Model_U2Net(3)
    model.load_weights(model_path)
    with rasterio.open(image_path) as dataset_image:
        transform = dataset_image.transform
        proj_str = (dataset_image.crs.to_string())
        gdf_tree = gp.read_file(shape_path)
        gdf_tree = gdf_tree.to_crs(proj_str)
        bound_list = [(geo_row.geometry.bounds) for index, geo_row in gdf_tree.iterrows()]
        bound_list_polygon= [bbbox(bound[0],bound[1],bound[2],bound[3]) for bound in bound_list]
        bound_list_polygon_ext = [extend_bound(Polygon(bound_poly)) for bound_poly in bound_list_polygon]
        gdf_tree['geometry'] = bound_list_polygon_ext
        color = (0,0,0)
        polygon_pixel = [(transform_geom_to_poly_px(polygon, transform)) for polygon in gdf_tree['geometry']]
        list_cnt = []
        count = 0  
        for polygon in tqdm.tqdm(polygon_pixel):
            try:
                xmin,ymin = int(np.amin(polygon, axis=0)[0]),int(np.amin(polygon, axis=0)[1])
                xmax,ymax = int(np.amax(polygon, axis=0)[0]),int(np.amax(polygon, axis=0)[1])
                width,height = int(xmax - xmin), int(ymax - ymin)
                image_predict = dataset_image.read(window=Window(xmin, ymin, width, height))[0:3].swapaxes(0, 1).swapaxes(1,2)
                new_image_width = new_image_height = max(width,height)
                x_center = (new_image_width - width) // 2
                y_center = (new_image_height - height) // 2
                result = np.full((new_image_height,new_image_width, 3), color, dtype=np.uint8)
                result[y_center:y_center+height, x_center:x_center+width] = image_predict
                result = cv2.resize(result,(INPUT_SIZE, INPUT_SIZE), interpolation = cv2.INTER_CUBIC)
                prediction = model.predict(result[np.newaxis,...]/255.)[0]
                true_mask = prediction.reshape((320,320))
                true_mask = (true_mask>0.5).astype(np.uint8)
                kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
                closing = cv2.morphologyEx(true_mask, cv2.MORPH_CLOSE, kernel3)
                opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel3)
                true_mask=opening
                new_image_width = new_image_height = max(width,height)
                true_mask = (cv2.resize(true_mask,(int(new_image_width), int(new_image_width)), interpolation = cv2.INTER_CUBIC)>0.5).astype(np.uint8)
                x_center = (new_image_width - width) // 2
                y_center = (new_image_height - height) // 2
                true_mask_result = true_mask[y_center:y_center+height, x_center:x_center+width]
                contours, hierarchy = cv2.findContours(true_mask_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours)>0:
                    cnt_result = contours[0]
                    _, radius_f = cv2.minEnclosingCircle(cnt_result)
                    for cnt in contours:
                        (x, y), radius = cv2.minEnclosingCircle(cnt)
                        if radius>radius_f:
                            radius_f = radius
                            cnt_result = cnt
                    cnt_result = cnt_result.reshape(-1, 2) + np.array([xmin, ymin])
                    list_cnt.append(cnt_result)
            except Exception as e:
                count = count+1
                # print(count)
        if len(list_cnt)>0:
            list_geo_polygon = [Polygon(transform_poly_px_to_geom(polygon, transform)) for polygon in list_cnt if len(polygon)>=3]
            df_polygon = pd.DataFrame({'geometry': list_geo_polygon})
            gdf_polygon = gp.GeoDataFrame(df_polygon, geometry='geometry', crs=proj_str)
            # gdf_tree['geometry'] = gdf_tree.geometry.centroid.buffer(list_geo_polygon)
            if out_path.endswith('.shp'):
                gdf_polygon.to_file(out_path)
        else:
            pass
    return True


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.compat.v1.Session(config=config))


    ####################### predict#########################
#     os.environ['CUDA_VISIBLE_DEVICES'] = ''   
    image_path = r"/mnt/Nam/ml-models/pretrain-models/building-footprint-master/src/test/google_download.tif"
    input_box_shp = r"/mnt/Nam/ml-models/pretrain-models/building-footprint-master/src/test/google_download_test.shp"
    output_building_shp = r"/mnt/Nam/ml-models/pretrain-models/building-footprint-master/src/test/google_download_test_ressult_final.shp"
    model_path =r'u2net.h5'
    predict_building(image_path, input_box_shp, output_building_shp,model_path)
    
    ##################################################### train u2net
    # bce = tf.keras.losses.BinaryCrossentropy()
    # #     my_model = DexiNed(3)
    # my_model = Model_U2Net(3)
    # # my_model = NetWork(3)
    # #     my_model.load_weights('/media/skymap/Learnning/Nam_work_space/model/u2nnnnnn.h5')

    # path3 = glob.glob('/media/skymap/Learnning/Nam_work_space/built/data/image/*.tif')
    # np.random.shuffle(path3)

    # data_train = path3[:int(len(path3)* 0.7)]
    # data_test = path3[int(len(path3)* 0.7):]

    # traindata = DataParser(data_train)
    # testdata = DataParser(data_test)

    # TRAIN_LOGDIR = '/media/skymap/Learnning/Nam_work_space/built/logs'
    # TRAIN_EPOCHS = 70
    # best_val_loss = 1
    # steps_per_epoch = len(traindata)
    # global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    # total_steps = TRAIN_EPOCHS * steps_per_epoch


    # validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # print(f'GPUs {gpus}')
    # if len(gpus) > 0:
    #     try: tf.config.experimental.set_memory_growth(gpus[0], True)
    #     except RuntimeError: pass

    # #     if os.path.exists(TRAIN_LOGDIR): shutil.rmtree(TRAIN_LOGDIR)
    # writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # #     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # for epoch in range(TRAIN_EPOCHS):
    #     for image_data, target in traindata:
    #         results = train_step(image_data, target)
    #         cur_step = (results[0] - 2) %steps_per_epoch + 1
    #         print("epoch:{:2.0f} step:{:5.0f}/{}, total_loss:{:7.4f}".format(epoch, cur_step, steps_per_epoch, results[1]))

    #     count, total_val = 0., 0
    #     for image_data, target in testdata:
    #         results = validate_step(image_data, target)
    #         count += 1
    #         total_val += results

    #     with validate_writer.as_default():
    #         tf.summary.scalar("validate_loss/total_val", total_val/count, step=epoch)
    #     validate_writer.flush()
    #     print("\n\ntotal_val_loss:{:7.4f}\n\n".format(total_val/count))

    #     if best_val_loss>total_val/count:
    #         save_directory = os.path.join("/media/skymap/Learnning/Nam_work_space/built/model", f"u2net.h5")
    #         my_model.save_weights(save_directory)
    #         best_val_loss = total_val/count


    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.0002)
    # for epoch in range(TRAIN_EPOCHS, TRAIN_EPOCHS+30):
    #     for image_data, target in traindata:
    #         results = train_step(image_data, target)
    #         cur_step = (results[0] - 2) %steps_per_epoch + 1
    #         print("epoch:{:2.0f} step:{:5.0f}/{}, total_loss:{:7.4f}".format(epoch, cur_step, steps_per_epoch, results[1]))

    #     count, total_val = 0., 0
    #     for image_data, target in testdata:
    #         results = validate_step(image_data, target)
    #         count += 1
    #         total_val += results

    #     with validate_writer.as_default():
    #         tf.summary.scalar("validate_loss/total_val", total_val/count, step=epoch)
    #     validate_writer.flush()
    #     print("\n\ntotal_val_loss:{:7.4f}\n\n".format(total_val/count))

    #     if best_val_loss>total_val/count:
    #         save_directory = os.path.join("/media/skymap/Learnning/Nam_work_space/built/model", f"u2net.h5")
    #         my_model.save_weights(save_directory)
    #         best_val_loss = total_val/count