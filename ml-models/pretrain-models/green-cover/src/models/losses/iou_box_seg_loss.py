import tensorflow as tf

def iou_box_coef(y_true, y_pred, mode='giou', dtype=tf.float32):
    
    """
    Inersection over Union (IoU) and generalized IoU coefficients for bounding boxes.
    
    iou_box_coef(y_true, y_pred, mode='giou', dtype=tf.float32)
    
    ----------
    Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I. and Savarese, S., 2019. 
    Generalized intersection over union: A metric and a loss for bounding box regression. 
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 658-666).
    
    ----------
    Input
        y_true: the target bounding box. 
        y_pred: the predicted bounding box.
        
        Elements of a bounding box should be organized as: [y_min, x_min, y_max, x_max].

        mode: 'iou' for IoU coeff (i.e., Jaccard index);
              'giou' for generalized IoU coeff.
        
        dtype: the data type of input tensors.
               Default is tf.float32.

    """
    
    zero = tf.convert_to_tensor(0.0, dtype)
    
    # subtrack bounding box coords
    ymin_true, xmin_true, ymax_true, xmax_true = tf.unstack(y_true, 4, axis=-1)
    ymin_pred, xmin_pred, ymax_pred, xmax_pred = tf.unstack(y_pred, 4, axis=-1)
    
    # true area
    w_true = tf.maximum(zero, xmax_true - xmin_true)
    h_true = tf.maximum(zero, ymax_true - ymin_true)
    area_true = w_true * h_true
    
    # pred area
    w_pred = tf.maximum(zero, xmax_pred - xmin_pred)
    h_pred = tf.maximum(zero, ymax_pred - ymin_pred)
    area_pred = w_pred * h_pred
    
    # intersections
    intersect_ymin = tf.maximum(ymin_true, ymin_pred)
    intersect_xmin = tf.maximum(xmin_true, xmin_pred)
    intersect_ymax = tf.minimum(ymax_true, ymax_pred)
    intersect_xmax = tf.minimum(xmax_true, xmax_pred)
    
    w_intersect = tf.maximum(zero, intersect_xmax - intersect_xmin)
    h_intersect = tf.maximum(zero, intersect_ymax - intersect_ymin)
    area_intersect = w_intersect * h_intersect
    
    # IoU
    area_union = area_true + area_pred - area_intersect
    iou = tf.math.divide_no_nan(area_intersect, area_union)
    
    if mode == "iou":
        
        return iou
    
    else:
        
        # encolsed coords
        enclose_ymin = tf.minimum(ymin_true, ymin_pred)
        enclose_xmin = tf.minimum(xmin_true, xmin_pred)
        enclose_ymax = tf.maximum(ymax_true, ymax_pred)
        enclose_xmax = tf.maximum(xmax_true, xmax_pred)
        
        # enclosed area
        w_enclose = tf.maximum(zero, enclose_xmax - enclose_xmin)
        h_enclose = tf.maximum(zero, enclose_ymax - enclose_ymin)
        area_enclose = w_enclose * h_enclose
        
        # generalized IoU
        giou = iou - tf.math.divide_no_nan((area_enclose - area_union), area_enclose)

        return giou

def iou_box(y_true, y_pred, mode='giou', dtype=tf.float32):
    """
    Inersection over Union (IoU) and generalized IoU losses for bounding boxes. 
    
    iou_box(y_true, y_pred, mode='giou', dtype=tf.float32)
    
    ----------
    Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I. and Savarese, S., 2019. 
    Generalized intersection over union: A metric and a loss for bounding box regression. 
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 658-666).
    
    ----------
    Input
        y_true: the target bounding box. 
        y_pred: the predicted bounding box.
        
        Elements of a bounding box should be organized as: [y_min, x_min, y_max, x_max].

        mode: 'iou' for IoU coeff (i.e., Jaccard index);
              'giou' for generalized IoU coeff.
        
        dtype: the data type of input tensors.
               Default is tf.float32.
        
    """
    
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.cast(y_pred, dtype)
    
    y_true = tf.cast(y_true, dtype)
    
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)

    return 1 - iou_box_coef(y_true, y_pred, mode=mode, dtype=dtype)

def iou_seg(y_true, y_pred, dtype=tf.float32):
    """
    Inersection over Union (IoU) loss for segmentation maps. 
    
    iou_seg(y_true, y_pred, dtype=tf.float32)
    
    ----------
    Rahman, M.A. and Wang, Y., 2016, December. Optimizing intersection-over-union in deep neural networks for 
    image segmentation. In International symposium on visual computing (pp. 234-244). Springer, Cham.
    
    ----------
    Input
        y_true: segmentation targets, c.f. `keras.losses.categorical_crossentropy`
        y_pred: segmentation predictions.
        
        dtype: the data type of input tensors.
               Default is tf.float32.
        
    """

    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.cast(y_pred, dtype)
    y_true = tf.cast(y_true, y_pred.dtype)

    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])

    area_intersect = tf.reduce_sum(tf.multiply(y_true_pos, y_pred_pos))
    
    area_true = tf.reduce_sum(y_true_pos)
    area_pred = tf.reduce_sum(y_pred_pos)
    area_union = area_true + area_pred - area_intersect
    
    return 1-tf.math.divide_no_nan(area_intersect, area_union)