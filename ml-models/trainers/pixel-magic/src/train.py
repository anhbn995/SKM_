import os
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from unetprocessor import UnetRunning
from params import HOSTED_ENDPOINT
import requests


class TrainingCallback(Callback):

    def __init__(self, task_id):
        super(TrainingCallback, self).__init__()
        self.task_id = task_id

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        payload = []
        for key in keys:
            payload.append({
                'key': key,
                'value': str(logs[key])
            })
        payload = {
            'task_id': self.task_id,
            'epoch': epoch + 1,
            'payload': payload
        }
        url = '{}/internal/training/report'.format(HOSTED_ENDPOINT)
        requests.post(url, json=payload)


class TrainerUnet(UnetRunning):
    def __init__(self, model_path, data_dir, unet_type, **kwargs):
        super().__init__(unet_type, **kwargs)
        self.model_path = model_path
        self.data_dir = data_dir
        self.batch_size = kwargs.get('batch_size') or 8
        self.epochs = kwargs.get('epochs') or 100
        # self.epochs = 100
        self.task_id = kwargs.get('task_id')

    def get_data(self, path_train, train=True):
        # print(path_train)
        # print(g)
        ids = next(os.walk(path_train + "images"))[2]
        X = np.zeros((len(ids), self.size, self.size,
                     self.numbands), dtype=np.float32)
        if train:
            y = np.zeros((len(ids), self.size, self.size,
                         self.label_count), dtype=np.float32)
        print('Getting images ... ')

        for n, id_ in enumerate(ids):
            # Load images
            x_img = rasterio.open(path_train + '/images/' + id_).read()
            x_img = np.transpose(x_img, (1, 2, 0))/255.0

            # Load masks
            if train:

                mask = rasterio.open(path_train + '/masks/' + id_).read()
                onehot = np.zeros((self.size, self.size, self.label_count))

                for idx, h in enumerate(self.labels):
                    onehot[:, :, idx+1] = (mask == h).astype(np.uint8)

                onehot[..., 0] = (mask == 1).astype(np.uint8)

            # Save images
            try:
                X[n, ...] = x_img.squeeze()
            except:
                X[n, ...] = x_img
            if train:
                try:
                    y[n, ...] = onehot.squeeze().astype(np.uint8)
                    # y.append(mask)
                except:
                    y[n, ...] = onehot.astype(np.uint8)
                    # y.append(mask)
        print('Done!')
        if train:
            return X, y
        else:
            return X

    def training(self, **kwargs):
        path_train = '{}/train/'.format(self.data_dir)
        X, y = self.get_data(path_train, train=True)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.15, random_state=2019)

        self.model = super().identify_model()

        if not os.path.exists(self.model_path):
            pass
        else:
            self.model.load_weights(self.model_path)

        patience_early = kwargs.get('patience_early') or 10
        factor = kwargs.get('factor') or 0.1
        patience_reduce = kwargs.get('patience_reduce') or 3
        min_lr = kwargs.get('min_lr') or 0.00001
        verbose = 1

        callbacks = [
            EarlyStopping(patience=patience_early, verbose=verbose),
            ReduceLROnPlateau(
                factor=factor, patience=patience_reduce, min_lr=min_lr, verbose=verbose),
            ModelCheckpoint(self.model_path, verbose=verbose,
                            save_best_only=True, save_weights_only=True)
        ]
        if self.task_id != -1:
            callbacks.append(TrainingCallback(self.task_id))
        results = self.model.fit(X_train, y_train, batch_size=int(self.batch_size), epochs=int(self.epochs), callbacks=callbacks,
                                 validation_data=(X_valid, y_valid))

        # from numba import cuda
        # cuda.select_device(0)
        # cuda.close()
        return {
            'optimizer': self.optimizer_code,
            'loss': self.loss,
            'metrics': self.metrics,
            'epochs': self.epochs,
            'img_size': self.size,
            'numbands': self.numbands,
            'label_count': self.label_count
        }
