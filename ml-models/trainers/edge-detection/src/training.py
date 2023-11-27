from __future__ import print_function
from keras import backend as K
from keras import callbacks
from model_architecture import hed, DataParser
import requests
from params import HOSTED_ENDPOINT


class TrainingCallback(callbacks.Callback):

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
        print(url)
        print(payload)
        request = requests.post(url, json=payload)
        print(request)


def train(train_data_dir, model_path, numbands=3, nepoch=100, learning_rate=0.00001, task_id=-1):
    initial_epoch = 0
    batch_size_train = 5

    K.set_image_data_format('channels_last')
    K.image_data_format()

    dataParser = DataParser(train_data_dir, batch_size_train)

    model = hed(numbands)

    K.set_value(model.optimizer.lr, learning_rate)

    checkpointer = callbacks.ModelCheckpoint(
        filepath=model_path, verbose=1, save_weights_only=True)

    callbacks = [checkpointer]
    if task_id != -1:
        callbacks.append(TrainingCallback(task_id))
    model.fit(
        dataParser.generate_minibatches(),
        initial_epoch=initial_epoch,
        steps_per_epoch=dataParser.steps_per_epoch,  # batch size
        epochs=initial_epoch + nepoch,
        validation_data=dataParser.generate_minibatches(train=False),
        validation_steps=dataParser.validation_steps,
        callbacks=callbacks
    )
