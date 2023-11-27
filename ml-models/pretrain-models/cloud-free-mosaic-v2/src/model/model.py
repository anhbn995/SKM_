from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, CSVLogger, History, EarlyStopping, LambdaCallback,ReduceLROnPlateau)
from tensorflow.keras import layers, backend, Model, utils

losses= tf.keras.losses.BinaryCrossentropy()
smooth = 1.

def BatchActivate(x):
    x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    x = layers.LeakyReLU()(x)
    return x

def convolution_block(x, n_filters, size, strides=(1,1), padding='same', activation=True):
    x = layers.Conv2D(n_filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, n_filters=16, batch_activate = False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, n_filters, (3,3))
    x = convolution_block(x, n_filters, (3,3))
    x = layers.Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x

def build_model(input_shape, n_filters, DropoutRatio=0.3):
    input_layer = tf.keras.Input(shape=input_shape)
    # 101 -> 50
    conv1 = layers.Conv2D(n_filters * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1, n_filters * 1)
    conv1 = residual_block(conv1, n_filters * 1, True)
    pool1 = layers.MaxPool2D((2, 2))(conv1)
    pool1 = layers.Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = layers.Conv2D(n_filters * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2, n_filters * 2)
    conv2 = residual_block(conv2, n_filters * 2, True)
    pool2 = layers.MaxPool2D((2, 2))(conv2)
    pool2 = layers.Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = layers.Conv2D(n_filters * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3, n_filters * 4)
    conv3 = residual_block(conv3, n_filters * 4, True)
    pool3 = layers.MaxPool2D((2, 2))(conv3)
    pool3 = layers.Dropout(DropoutRatio)(pool3)

    # Middle
    convm = layers.Conv2D(n_filters * 8, (3, 3), activation=None, padding="same")(pool3)
    convm = residual_block(convm, n_filters * 8)
    convm = residual_block(convm, n_filters * 8, True)

    # 12 -> 25
    deconv3 = layers.Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(DropoutRatio)(uconv3)

    uconv3 = layers.Conv2D(n_filters * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, n_filters * 4)
    uconv3 = residual_block(uconv3, n_filters * 4, True)

    # 25 -> 50
    deconv2 = layers.Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])

    uconv2 = layers.Dropout(DropoutRatio)(uconv2)
    uconv2 = layers.Conv2D(n_filters * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, n_filters * 2)
    uconv2 = residual_block(uconv2, n_filters * 2, True)

    # 50 -> 101
    deconv1 = layers.Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])

    uconv1 = layers.Dropout(DropoutRatio)(uconv1)
    uconv1 = layers.Conv2D(n_filters * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, n_filters * 1)
    uconv1 = residual_block(uconv1, n_filters * 1, True)

    # uconv1 = Dropout(DropoutRatio/2)(uconv1)
    output_layer_noActi = layers.Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = layers.Activation('sigmoid')(output_layer_noActi)
    
    model = Model(inputs=[input_layer], outputs=[output_layer])
    # optimizer=tf.keras.optimizers.Adam(learning_rate = 3e-04, clipvalue = 0.5)
    optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-05, momentum = 0.9)

    from tensorflow.keras import metrics
    model.compile(optimizer=optimizer,
                loss=losses,
                # metrics=['accuracy', f1_m, precision_m, recall_m]
                  metrics=['accuracy', f1_m, precision_m, recall_m, dice_coef]
    )
    return model

def recall_m(y_true, y_pred):
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + backend.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + backend.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+backend.epsilon()))

def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def cross_entropy_balanced(y_true, y_pred):
    _epsilon = _to_tensor(tf.keras.backend.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.math.log(y_pred/ (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)
    beta = count_neg / (count_neg + count_pos)
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)
    cost = tf.reduce_mean(cost * (1 - beta))

    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)

def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f * y_true_f) + tf.keras.backend.sum(y_pred_f * y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred) #+ cross_entropy_balanced(y_true, y_pred) # + losses(y_true, y_pred)

def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    inse = tf.math.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.math.reduce_sum(output * output, axis=axis)
        r = tf.math.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.math.reduce_sum(output, axis=axis)
        r = tf.math.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice

def binary_focal_loss_fixed(y_true, y_pred):
    gamma=2.
    alpha=.25
    y_true = tf.cast(y_true, tf.float32)
    epsilon = backend.epsilon()
    y_pred = backend.clip(y_pred, epsilon, 1.0 - epsilon)

    p_t = tf.where(backend.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_factor = backend.ones_like(y_true) * alpha

    alpha_t = tf.where(backend.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    cross_entropy = -backend.log(p_t)
    weight = alpha_t * backend.pow((1 - p_t), gamma)
    loss = weight * cross_entropy
    loss = backend.mean(backend.sum(loss, axis=1))
    return loss