import tensorflow as tf

def ms_ssim(y_true, y_pred, **kwargs):
    """
    Multiscale structural similarity (MS-SSIM) loss.
    
    ms_ssim(y_true, y_pred, **tf_ssim_kw)
    
    ----------
    Wang, Z., Simoncelli, E.P. and Bovik, A.C., 2003, November. Multiscale structural similarity for image quality assessment. 
    In The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003 (Vol. 2, pp. 1398-1402). Ieee.
    
    ----------
    Input
        kwargs: keywords of `tf.image.ssim_multiscale`
                https://www.tensorflow.org/api_docs/python/tf/image/ssim_multiscale
                
        *Issues of `tf.image.ssim_multiscale`refers to:
                https://stackoverflow.com/questions/57127626/error-in-calculation-of-inbuilt-ms-ssim-function-in-tensorflow
    
    """
    
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    
    tf_ms_ssim = tf.image.ssim_multiscale(y_true, y_pred, **kwargs)
        
    return 1 - tf_ms_ssim