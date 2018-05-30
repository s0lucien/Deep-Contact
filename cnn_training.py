from __future__ import absolute_import, division, print_function

import numpy as np
from keras import losses
from keras import optimizers
import keras.backend as K

from visual_deep_learning.cnn_model import learning_model
from src.gen_data.load_grid import load_grid


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-p', '--path', dest='path')
    parser.add_option('-n', '--number', type='int', dest='num')

    options, _ = parser.parse_args()
    path = options.path
    number = options.num

    model = learning_model(
        batch_size=1000,
        metrics=[losses.mean_absolute_percentage_error],
        optimizer=optimizers.SGD(
            lr=0.01, decay=1e-6, momentum=0.9, nesterov=True,
        ),
        log_dir='./log/',
        loss_func=losses.mean_absolute_percentage_error,
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
