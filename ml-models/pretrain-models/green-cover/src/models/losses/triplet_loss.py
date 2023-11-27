import tensorflow as tf

def triplet_1d(y_true, y_pred, N, margin=5.0):
    
    '''
    (Experimental)
    Semi-hard triplet loss with one-dimensional vectors of anchor, positive, and negative.
    
    triplet_1d(y_true, y_pred, N, margin=5.0)
    
    Input
    ----------
        y_true: a dummy input, not used within this function. Appeared as a requirment of tf.keras.loss function format.
        y_pred: a single pass of triplet training, with `shape=(batch_num, 3*embeded_vector_size)`.
                i.e., `y_pred` is the ordered and concatenated anchor, positive, and negative embeddings.
        N: Size (dimensions) of embedded vectors
        margin: a positive number that prevents negative loss.
        
    '''
    
    # anchor sample pair separations.
    Embd_anchor = y_pred[:, 0:N]
    Embd_pos = y_pred[:, N:2*N]
    Embd_neg = y_pred[:, 2*N:]
    
    # squared distance measures
    d_pos = tf.reduce_sum(tf.square(Embd_anchor - Embd_pos), 1)
    d_neg = tf.reduce_sum(tf.square(Embd_anchor - Embd_neg), 1)
    loss_val = tf.maximum(0., margin + d_pos - d_neg)
    loss_val = tf.reduce_mean(loss_val)
    
    return loss_val