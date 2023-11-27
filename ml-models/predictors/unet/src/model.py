from keras.models import Model
from keras.layers import Input, Activation, Dropout, Concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Convolution2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers import concatenate as merge_l

from utils.unet import *


class ArchitechUnet:
    def __init__(self, **kwargs):
        self.numbands = kwargs.get('numbands') or 4
        self.size = kwargs.get('trainer_size') or 128
        count = kwargs.get('count')
        if not count:
            self.label_count = len(kwargs.get('labels')) + 1
        else:
            self.label_count = count
        self.labels = list(
            map(lambda el: el.get('value'), kwargs.get('labels')))

        self.optimizer_code = kwargs.get('optimizer') or 'adam'
        self.optimizer = code_2_optimizer(self.optimizer_code)
        self.loss = kwargs.get('loss') or "categorical_crossentropy"
        self.metrics = kwargs.get('metrics') or ["accuracy"]
        self.n_filters = kwargs.get('n_filters') or 16
        self.dropout = kwargs.get('dropout') or 0.5
        self.batchnorm = kwargs.get('batchnorm') or True

    def get_unet(self):

        input_img = Input((self.size, self.size, self.numbands), name='img')
        # contracting path
        c1 = conv2d_block(input_img, n_filters=self.n_filters * 1,
                          kernel_size=3, batchnorm=self.batchnorm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(self.dropout * 0.5)(p1)

        c2 = conv2d_block(p1, n_filters=self.n_filters * 2,
                          kernel_size=3, batchnorm=self.batchnorm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(self.dropout)(p2)

        c3 = conv2d_block(p2, n_filters=self.n_filters * 4,
                          kernel_size=3, batchnorm=self.batchnorm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(self.dropout)(p3)

        c4 = conv2d_block(p3, n_filters=self.n_filters * 8,
                          kernel_size=3, batchnorm=self.batchnorm)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)
        p4 = Dropout(self.dropout)(p4)

        c5 = conv2d_block(p4, n_filters=self.n_filters * 16,
                          kernel_size=3, batchnorm=self.batchnorm)

        # expansive path
        u6 = Conv2DTranspose(self.n_filters * 8, (3, 3),
                             strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(self.dropout)(u6)
        c6 = conv2d_block(u6, n_filters=self.n_filters * 8,
                          kernel_size=3, batchnorm=self.batchnorm)

        u7 = Conv2DTranspose(self.n_filters * 4, (3, 3),
                             strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(self.dropout)(u7)
        c7 = conv2d_block(u7, n_filters=self.n_filters * 4,
                          kernel_size=3, batchnorm=self.batchnorm)

        u8 = Conv2DTranspose(self.n_filters * 2, (3, 3),
                             strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(self.dropout)(u8)
        c8 = conv2d_block(u8, n_filters=self.n_filters * 2,
                          kernel_size=3, batchnorm=self.batchnorm)

        u9 = Conv2DTranspose(self.n_filters * 1, (3, 3),
                             strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        u9 = Dropout(self.dropout)(u9)
        c9 = conv2d_block(u9, n_filters=self.n_filters * 1,
                          kernel_size=3, batchnorm=self.batchnorm)

        outputs = Conv2D(self.label_count, (1, 1), activation='softmax')(c9)
        model = Model(inputs=[input_img], outputs=[outputs])
        model.compile(self.optimizer, self.loss, self.metrics)

        return model

    def get_unet50(self):
        inputs = Input((self.size, self.size, self.numbands), name='img')
        [C1, C2, C3, C4, C5] = resnet_graph(
            inputs, "resnet50", stage5=True, train_bn=True)
        skip_layer = [C4, C3, C2, C1]
        n_upsample_blocks = 5
        upsample_rates = (2, 2, 2, 2, 2)
        decoder_filters = 64
        last_block_filters = decoder_filters
        up_block = Transpose2D_block
        x = skip_layer[0]
        for i in range(1, n_upsample_blocks):
            if i < 3:
                skip = skip_layer[i]
            else:
                skip = None
            up_size = (upsample_rates[i], upsample_rates[i])
            filters = last_block_filters * 2**(n_upsample_blocks-(i+1))
            x = up_block(filters, i, upsample_rate=up_size, skip=skip)(x)
        if self.label_count < 2:
            activation = 'sigmoid'
        else:
            activation = 'softmax'

        x = Conv2D(self.label_count, (3, 3),
                   padding='same', name='final_conv')(x)
        output = Activation(activation, name=activation)(x)

        model = Model(input=inputs, output=output)

        optimizer = SGD(lr=1e-3, decay=1e-8, momentum=0.9, nesterov=True)
        model.compile(optimizer=optimizer,
                      loss=dice_coef_loss2,
                      metrics=['accuracy', jaccard_coef, jaccard_coef_int])

        return model

    def get_unetfarm(self):
        # Input
        # img_input = Input(shape=(480,480,4), name='input')
        img_input = Input(shape=(self.size, self.size,
                          self.numbands), name='input')

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same',
                   name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu',
                   padding='same', name='block1_conv2')(x)
        b1 = side_branch(x, 1)  # 480 480 1
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                         name='block1_pool')(x)  # 240 240 64

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu',
                   padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu',
                   padding='same', name='block2_conv2')(x)
        b2 = side_branch(x, 2)  # 480 480 1
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                         name='block2_pool')(x)  # 120 120 128

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu',
                   padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu',
                   padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu',
                   padding='same', name='block3_conv3')(x)
        b3 = side_branch(x, 4)  # 480 480 1
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                         name='block3_pool')(x)  # 60 60 256

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu',
                   padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu',
                   padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu',
                   padding='same', name='block4_conv3')(x)
        b4 = side_branch(x, 8)  # 480 480 1
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same',
                         name='block4_pool')(x)  # 30 30 512

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu',
                   padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu',
                   padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same',
                   name='block5_conv3')(x)  # 30 30 512
        b5 = side_branch(x, 16)  # 480 480 1

        # fuse
        fuse = Concatenate(axis=-1)([b1, b2, b3, b4, b5])
        fuse = Conv2D(1, (1, 1), padding='same', use_bias=False,
                      activation=None)(fuse)  # 480 480 1

        # outputs
        o1 = Activation('sigmoid', name='o1')(b1)
        o2 = Activation('sigmoid', name='o2')(b2)
        o3 = Activation('sigmoid', name='o3')(b3)
        o4 = Activation('sigmoid', name='o4')(b4)
        o5 = Activation('sigmoid', name='o5')(b5)
        ofuse = Activation('sigmoid', name='ofuse')(fuse)

        # model
        model = Model(inputs=[img_input], outputs=[o1, o2, o3, o4, o5, ofuse])
        # filepath = r"F:\data\farm\codes\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
        # load_weights_from_hdf5_group_by_name(model, filepath)

        model.compile(loss={'o1': cross_entropy_balanced,
                            'o2': cross_entropy_balanced,
                            'o3': cross_entropy_balanced,
                            'o4': cross_entropy_balanced,
                            'o5': cross_entropy_balanced,
                            'ofuse': cross_entropy_balanced,
                            },
                      #   metrics={'ofuse': ofuse_pixel_error},
                      metrics={'ofuse': ofuse_pixel_accuracy},
                      optimizer='adam')

        return model

    def unet_basic(self):
        conv_params = dict(activation='relu', border_mode='same')
        merge_params = dict(axis=-1)
        inputs1 = Input((self.size, self.size, self.numbands))
        # inputs2 = Input((size, size,int(num_channel)))
        # merge_input = concatenate([inputs1, inputs2])
        conv1 = Convolution2D(32, (3, 3), **conv_params)(inputs1)
        conv1 = Convolution2D(32, (3, 3), **conv_params)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Convolution2D(64, (3, 3), **conv_params)(pool1)
        conv2 = Convolution2D(64, (3, 3), **conv_params)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Convolution2D(128, (3, 3), **conv_params)(pool2)
        conv3 = Convolution2D(128, (3, 3), **conv_params)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Convolution2D(256, (3, 3), **conv_params)(pool3)
        conv4 = Convolution2D(256, (3, 3), **conv_params)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Convolution2D(512, (3, 3), **conv_params)(pool4)
        conv5 = Convolution2D(512, (3, 3), **conv_params)(conv5)

        up6 = merge_l([UpSampling2D(size=(2, 2))
                      (conv5), conv4], **merge_params)
        conv6 = Convolution2D(256, (3, 3), **conv_params)(up6)
        conv6 = Convolution2D(256, (3, 3), **conv_params)(conv6)

        up7 = merge_l([UpSampling2D(size=(2, 2))
                      (conv6), conv3], **merge_params)
        conv7 = Convolution2D(128, (3, 3), **conv_params)(up7)
        conv7 = Convolution2D(128, (3, 3), **conv_params)(conv7)

        up8 = merge_l([UpSampling2D(size=(2, 2))
                      (conv7), conv2], **merge_params)
        conv8 = Convolution2D(64, (3, 3), **conv_params)(up8)
        conv8 = Convolution2D(64, (3, 3), **conv_params)(conv8)

        up9 = merge_l([UpSampling2D(size=(2, 2))
                      (conv8), conv1], **merge_params)
        conv9 = Convolution2D(32, (3, 3), **conv_params)(up9)
        conv9 = Convolution2D(32, (3, 3), **conv_params)(conv9)

        conv10 = Convolution2D(1, (1, 1), activation='sigmoid')(conv9)
        optimizer = Adam(lr=1e-3)
        # optimizer=SGD(lr=1e-3, decay=1e-8, momentum=0.9, nesterov=True)
        model = Model(input=inputs1, output=conv10)
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy', jaccard_coef, jaccard_coef_int])
        return model
