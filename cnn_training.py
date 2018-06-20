from __future__ import absolute_import, division, print_function

from optparse import OptionParser

parser = OptionParser()
parser.add_option('-p', '--path', dest='path')
parser.add_option('-n', '--number', type='int', dest='num')
parser.add_option('-g', '--gpu', type='str', dest='gpu', default='0')

options, _ = parser.parse_args()

# using GPU for training
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

import numpy as np
import datetime
from keras import optimizers
import keras.backend as K

from visual_deep_learning.cnn_model import (
    learning_model,
    loss_func,
    mean_absolute_loss,
)
from src.gen_data.load_grid import load_grid


if __name__ == '__main__':
    path = options.path
    number = options.num

    current_date = datetime.datetime.now().strftime('%Y-%m-%d')

    model = learning_model(
        batch_size=200,
        epochs=1000,
        metrics=[
            loss_func,
            mean_absolute_loss],
        optimizer=optimizers.SGD(
            lr=0.01, decay=1e-6, momentum=0.9, nesterov=True,
        ),
        log_dir='./log/{}'.format(current_date),
        loss_func=loss_func,
    )
    x_tr = np.concatenate([
        load_grid(path, num)[0]
        for num in range(number)
    ])
    y_tr = np.concatenate([
        load_grid(path, num)[1]
        for num in range(number)
    ])

    model.train(x_tr, y_tr, method='batch_method')
