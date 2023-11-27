import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

def _crps_tf(y_true, y_pred, factor=0.05):
    
    '''
    core of (pseudo) CRPS loss.
    
    y_true: two-dimensional arrays
    y_pred: two-dimensional arrays
    factor: importance of std term
    '''
    
    # mean absolute error
    mae = K.mean(tf.abs(y_pred - y_true))
    
    dist = tf.math.reduce_std(y_pred)
    
    return mae - factor*dist

def crps2d_tf(y_true, y_pred, factor=0.05):
    
    '''
    (Experimental)
    An approximated continuous ranked probability score (CRPS) loss function:
    
        CRPS = mean_abs_err - factor * std
        
    * Note that the "real CRPS" = mean_abs_err - mean_pairwise_abs_diff
    
     Replacing mean pairwise absolute difference by standard deviation offers
     a complexity reduction from O(N^2) to O(N*logN) 
    
    ** factor > 0.1 may yield negative loss values.
    
    Compatible with high-level Keras training methods
    
    Input
    ----------
        y_true: training target with shape=(batch_num, x, y, 1)
        y_pred: a forward pass with shape=(batch_num, x, y, 1)
        factor: relative importance of standard deviation term.
        
    '''
    
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    
    batch_num = y_pred.shape.as_list()[0]
    
    crps_out = 0
    for i in range(batch_num):
        crps_out += _crps_tf(y_true[i, ...], y_pred[i, ...], factor=factor)
        
    return crps_out/batch_num


def _crps_np(y_true, y_pred, factor=0.05):
    
    '''
    Numpy version of _crps_tf.
    '''
    
    # mean absolute error
    mae = np.nanmean(np.abs(y_pred - y_true))
    dist = np.nanstd(y_pred)
    
    return mae - factor*dist

def crps2d_np(y_true, y_pred, factor=0.05):
    
    '''
    (Experimental)
    Nunpy version of `crps2d_tf`.
    
    Documentation refers to `crps2d_tf`.
    '''
    
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    
    batch_num = len(y_pred)
    
    crps_out = 0
    for i in range(batch_num):
        crps_out += _crps_np(y_true[i, ...], y_pred[i, ...], factor=factor)
        
    return crps_out/batch_num