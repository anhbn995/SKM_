import tensorflow as tf
from unet_models.losses.iou_box_seg_loss import iou_seg
from unet_models.losses.tversky_loss import focal_tversky

def hybrid_loss(y_true, y_pred):

    loss_focal = focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3)
    loss_iou = iou_seg(y_true, y_pred)
    
    # (x) 
    #loss_ssim = losses.ms_ssim(y_true, y_pred, max_val=1.0, filter_size=4)
    
    return loss_focal+loss_iou #+loss_ssim

@tf.function
def segmentation_loss(y_true, y_pred, weight=None, N_CLASSES=4):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    cross_entropy_loss = cce(y_true=y_true, y_pred=y_pred)
    dice_loss = gen_dice(y_true, y_pred, N_CLASSES)
    return 0.5 * cross_entropy_loss + 0.5 * dice_loss

@tf.function
def dice_per_class(y_true, y_pred, eps=1e-5):
    intersect = tf.reduce_sum(y_true * y_pred)
    y_sum = tf.reduce_sum(y_true * y_true)
    z_sum = tf.reduce_sum(y_pred * y_pred)
    loss = 1 - (2 * intersect + eps) / (z_sum + y_sum + eps)
    return loss

@tf.function
def gen_dice(y_true, y_pred, weight=None, N_CLASSES=2):
    """both tensors are [b, h, w, classes] and y_pred is in logit form"""
    # [b, h, w, classes]
    pred_tensor = tf.nn.softmax(y_pred)
    if weight is None:
            weight = [1] * N_CLASSES
    # assert pred_tensor.shape == y_true.shape, 'predict {} & target {} shape do not match'.format(pred_tensor.shape, y_true.shape)
    loss = 0.0
    for c in range(N_CLASSES):
        loss += dice_per_class(y_true[:, :, :, c], pred_tensor[:, :, :, c])*weight
    return loss/N_CLASSES

