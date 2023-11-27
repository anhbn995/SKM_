import numpy as np
import rasterio
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras

def random_rotate(image, label):
    rot = np.random.uniform(-20 * np.pi / 180, 20 * np.pi / 180)
    modified = tfa.image.rotate(image, rot)
    m_label = tfa.image.rotate(label, rot)
    return modified, m_label

def random_rot_flip(image, label, width, height):
    m_label = tf.reshape(label, (width, height, 1))
    axis = np.random.randint(0, 2)
    if axis == 1:
        # vertical flip
        modified = tf.image.flip_left_right(image=image)
        m_label = tf.image.flip_left_right(image=m_label)
    else:
        # horizontal flip
        modified = tf.image.flip_up_down(image=image)
        m_label = tf.image.flip_up_down(image=m_label)
    # rot 90
    k_90 = np.random.randint(4)
    modified = tf.image.rot90(image=modified, k=k_90)
    m_label = tf.image.rot90(image=m_label, k=k_90)

    m_label = tf.reshape(m_label, (width, height))
    return modified, m_label


def data_augment(image, label, N_CLASSES):
    rand1, rand2 = np.random.uniform(size=(2, 1))
    w, h = image.shape[0], image.shape[1]
    if rand1 > 0.25:
        modified, m_label = random_rot_flip(image, label, w, h)
    elif rand2 > 0.25:
        modified, m_label = random_rotate(image, label)
    else:
        modified, m_label = image, label
    m_label = tf.cast(m_label, tf.uint8)
    m_label = tf.one_hot(m_label, depth=N_CLASSES)
    return modified, m_label


def dummy_loader(model_path):
    '''
    Load a stored keras model and return its weights.
    
    Input
    ----------
        The file path of the stored keras model.
    
    Output
    ----------
        Weights of the model.
        
    '''
    backbone = keras.models.load_model(model_path, compile=False)
    W = backbone.get_weights()
    return W

# def image_to_array(filenames, size, channel):
#     '''
#     Converting RGB images to numpy arrays.
    
#     Input
#     ----------
#         filenames: an iterable of the path of image files
#         size: the output size (height == width) of image. 
#               Processed through PIL.Image.NEAREST
#         channel: number of image channels, e.g. channel=3 for RGB.
        
#     Output
#     ----------
#         An array with shape = (filenum, size, size, channel)
        
#     '''
    
#     # number of files
#     L = len(filenames)
    
#     # allocation
#     out = np.empty((L, size, size, channel))
    
#     # loop over filenames
#     if channel == 1:
#         for i, name in enumerate(filenames):
#             with Image.open(name) as pixio:
#                 pix = pixio.resize((size, size), Image.NEAREST)
#                 out[i, ..., 0] = np.array(pix)
#     else:
#         for i, name in enumerate(filenames):
#             with Image.open(name) as pixio:
#                 pix = pixio.resize((size, size), Image.NEAREST)
#                 out[i, ...] = np.array(pix)[..., :channel]
#     return out[:, ::-1, ...]

def image_to_array(filenames, size, channel):
    # number of files
    L = len(filenames)
    
    # allocation
    out = np.empty((L, size, size, channel))
    
    # loop over filenames
    if channel == 1:
        for i, name in enumerate(filenames):
            with rasterio.open(name) as pixio:
                pix = pixio.read().swapaxes(0,1).swapaxes(1,2)
                out[i, ...] = np.array(pix)
    else:
        for i, name in enumerate(filenames):            
            with rasterio.open(name) as pixio:
                pix = pixio.read().swapaxes(0,1).swapaxes(1,2)
                out[i, ...] = np.array(pix)[..., :channel]
    return out[:, ::-1, ...]

def shuffle_ind(L):
    '''
    Generating random shuffled indices.
    
    Input
    ----------
        L: an int that defines the largest index
        
    Output
    ----------
        a numpy array of shuffled indices with shape = (L,)
    '''
    
    ind = np.arange(L)
    np.random.shuffle(ind)
    return ind

def freeze_model(model, freeze_batch_norm=False):
    '''
    freeze a keras model
    
    Input
    ----------
        model: a keras model
        freeze_batch_norm: False for not freezing batch notmalization layers
    '''
    if freeze_batch_norm:
        for layer in model.layers:
            layer.trainable = False
    else:
        from tensorflow.keras.layers import BatchNormalization    
        for layer in model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
    return model


