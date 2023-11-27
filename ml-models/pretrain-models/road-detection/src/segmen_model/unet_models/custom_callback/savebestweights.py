import numpy as np
import tensorflow as tf

class SavebestweightsandEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=0, weights_path='./weights/unet3plus.h5', restore=False):
        # super(SavebestweightsandEarlyStopping, self).__init__()
        self.patience = patience
        self.best_weights = None
        self.weights_path = weights_path
        self.restore = restore

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
            self.model.save_weights(self.weights_path)
            print("Save best weights.")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore:
                    print("Restoring model weights from the end of the best epoch.")
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

