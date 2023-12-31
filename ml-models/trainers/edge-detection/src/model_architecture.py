import os
import numpy as np
import glob
import cv2
import tensorflow as tf

from keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D
from keras.layers import Concatenate, Activation
from keras.models import Model
from keras import backend as K


class DataParser():

    def __init__(self, train_data_dir, batch_size_train):
        self.train_data_dir = train_data_dir
        filelist = glob.glob(os.path.join(self.train_data_dir, 'images/*.npz'))
        self.samples = [(f, f.replace('images', 'masks')) for f in filelist]

        self.n_samples = len(self.samples)
        self.all_ids = list(range(self.n_samples))
        np.random.shuffle(self.all_ids)

        train_split = 0.8
        self.training_ids = self.all_ids[:int(train_split * self.n_samples)]
        self.validation_ids = self.all_ids[int(train_split * self.n_samples):]

        self.batch_size_train = batch_size_train
        self.steps_per_epoch = int(len(self.training_ids)//batch_size_train)

        self.validation_steps = int(
            len(self.validation_ids)//(batch_size_train*2))

        # self.augmentations = [self.flip_ud, self.flip_lr, self.rot90, self.blur]
        self.augmentations = [self.flip_ud, self.flip_lr, self.rot90]

    def generate_minibatches(self, train=True):
        while True:
            if train:
                np.random.shuffle(self.training_ids)
                for i in range(self.steps_per_epoch):
                    batch_ids = self.training_ids[i *
                                                  self.batch_size_train:(i+1)*self.batch_size_train]

                    ims, ems, _ = self.get_batch(batch_ids)
                    yield(ims, [ems, ems, ems, ems, ems, ems])

            else:
                np.random.shuffle(self.validation_ids)
                for i in range(self.validation_steps):
                    batch_ids = self.validation_ids[i*self.batch_size_train*2:(
                        i+1)*self.batch_size_train*2]

                    ims, ems, _ = self.get_batch(batch_ids)
                    yield(ims, [ems, ems, ems, ems, ems, ems])

    def get_batch(self, batch):
        images = []
        edgemaps = []
        filenames = []

        for idx, b in enumerate(batch):

            with np.load(self.samples[b][0]) as f:
                im = f['arr']
            with np.load(self.samples[b][1]) as f:
                em = f['arr']

            im = np.array(im/255.0, dtype=np.float32)
            em = np.array(np.expand_dims(em, axis=2), dtype=np.float32)

            for f in self.augmentations:
                if np.random.uniform() < 0.20:
                    im, em = f(im, em)

            images.append(im)
            edgemaps.append(em)
            filenames.append(self.samples[b])

        images = np.asarray(images)
        edgemaps = np.asarray(edgemaps)

        return images, edgemaps, filenames

    def flip_ud(self, im, em):
        return np.flipud(im), np.flipud(em)

    def flip_lr(self, im, em):
        return np.fliplr(im), np.fliplr(em)

    def rot90(self, im, em):
        return np.rot90(im), np.rot90(em)

    def blur(self, im, em):
        return cv2.GaussianBlur(im, (5, 5), 0), em


def side_branch(x, factor):
    x = Conv2D(1, (1, 1), activation=None, padding='same')(x)

    kernel_size = (2*factor, 2*factor)
    x = Conv2DTranspose(1, kernel_size, strides=factor,
                        padding='same', use_bias=False, activation=None)(x)

    return x


def hed(image_band_channel):
    # Input
    img_input = Input(shape=(480, 480, image_band_channel), name='input')

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


def cross_entropy_balanced(y_true, y_pred):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(
        logits=y_pred, labels=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def ofuse_pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')


def ofuse_pixel_accuracy(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    acc = tf.cast(tf.equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(acc, name='pixel_accuracy')


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def load_weights_from_hdf5_group_by_name(model, filepath):
    ''' Name-based weight loading '''

    import h5py

    f = h5py.File(filepath, mode='r')

    flattened_layers = model.layers
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in flattened_layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    # we batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]

        for layer in index.get(name, []):
            symbolic_weights = layer.weights
            if len(weight_values) != len(symbolic_weights):
                raise Exception('Layer #' + str(k) +
                                ' (named "' + layer.name +
                                '") expects ' +
                                str(len(symbolic_weights)) +
                                ' weight(s), but the saved weights' +
                                ' have ' + str(len(weight_values)) +
                                ' element(s).')
            # set values
            for i in range(len(weight_values)):
                weight_value_tuples.append(
                    (symbolic_weights[i], weight_values[i]))
                K.batch_set_value(weight_value_tuples)
