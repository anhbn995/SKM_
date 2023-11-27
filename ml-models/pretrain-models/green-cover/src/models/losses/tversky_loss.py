import tensorflow as tf
import tensorflow.keras.backend as K

def tversky_coef(y_true, y_pred, alpha=0.5, const=K.epsilon()):
    '''
    Weighted Sørensen–Dice coefficient.
    
    Input
    ----------
        y_true, y_pred: predicted outputs and targets.
        const: a constant that smooths the loss gradient and reduces numerical instabilities.
        
    '''
    
    # flatten 2-d tensors
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])
    
    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos  = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))
    false_pos = tf.reduce_sum((1-y_true_pos) * y_pred_pos)
    
    # TP/(TP + a*FN + b*FP); a+b = 1
    coef_val = (true_pos + const)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + const)
    
    return coef_val

def tversky(y_true, y_pred, alpha=0.5, const=K.epsilon()):
    '''
    Tversky Loss.
    
    tversky(y_true, y_pred, alpha=0.5, const=K.epsilon())
    
    ----------
    Hashemi, S.R., Salehi, S.S.M., Erdogmus, D., Prabhu, S.P., Warfield, S.K. and Gholipour, A., 2018. 
    Tversky as a loss function for highly unbalanced image segmentation using 3d fully convolutional deep networks. 
    arXiv preprint arXiv:1803.11078.
    
    Input
    ----------
        alpha: tunable parameter within [0, 1]. Alpha handles imbalance classification cases.
        const: a constant that smooths the loss gradient and reduces numerical instabilities.
        
    '''
    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    # <--- squeeze-out length-1 dimensions.
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    
    loss_val = 1 - tversky_coef(y_true, y_pred, alpha=alpha, const=const)
    
    return loss_val

def focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3, const=K.epsilon()):
    
    '''
    Focal Tversky Loss (FTL)
    
    focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3)
    
    ----------
    Abraham, N. and Khan, N.M., 2019, April. A novel focal tversky loss function with improved 
    attention u-net for lesion segmentation. In 2019 IEEE 16th International Symposium on Biomedical Imaging 
    (ISBI 2019) (pp. 683-687). IEEE.
    
    ----------
    Input
        alpha: tunable parameter within [0, 1]. Alpha handles imbalance classification cases 
        gamma: tunable parameter within [1, 3].
        const: a constant that smooths the loss gradient and reduces numerical instabilities.
        
    '''
    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    # <--- squeeze-out length-1 dimensions.
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    
    # (Tversky loss)**(1/gamma) 
    loss_val = tf.math.pow((1-tversky_coef(y_true, y_pred, alpha=alpha, const=const)), 1/gamma)
    
    return loss_val
