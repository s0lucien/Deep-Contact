from __future__ import absolute_import, division, print_function

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D,
)
from keras import optimizers
from keras.callbacks import (
    LearningRateScheduler,
    TensorBoard,
)
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model


class learning_model(object):
    def __init__(
        self,
        log_dir,
        # loss function, this is not classify question,
        # so choose another loss func here.
        # I think `mean squared error` should be good here.
        optimizer,
        loss_func,
        metrics=['accuracy'],
        batch_size=100,
        iterations=1000,
        epochs=1000,
        dropout=0.25,
        weight_decay=0.0001,
    ):
        '''
        :param input_shape:
            loss_func:
            metrics:
        '''
        self.model = Sequential()
        self.optimizer = optimizer
        self.iterations = iterations
        self.loss_func = loss_func
        self.metrics = metrics
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.weight_decay = weight_decay

    def build_model(self, input_shape, output_shape):
        self.model.add(Conv2D(64,
                              (3, 3),
                              padding='same',
                              kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                              kernel_initializer='he_normal',
                              input_shape=input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same'))

        self.model.add(Conv2D(128,
                              (3, 3),
                              padding='same',
                              kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                              kernel_initializer='he_normal'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same'))

        self.model.add(Conv2D(256,
                              (3, 3),
                              padding='same',
                              kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                              kernel_initializer='he_normal'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same'))

        self.model.add(Conv2D(512,
                              (3, 3),
                              padding='same',
                              kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                              kernel_initializer='he_normal'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding='same'))

        self.model.add(Dropout(self.dropout))
        self.model.add(Flatten())
        self.model.add(Dense(4000))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        # Flatten output here
        output_size = output_shape[0]

        self.model.add(Dense(output_size))
        self.model.add(Activation('relu'))

        self.model.compile(
            loss=self.loss_func,
            optimizer=self.optimizer,
            metrics=self.metrics,
        )

    def train(self, x_train, y_train, method, validation_rate=0.2, save=True):
        input_shape = x_train.shape[1:]
        output_shape = y_train.shape[1:]

        # set call_back
        tb_cb = TensorBoard(log_dir=self.log_dir)

        # change learning rate
        change_lr = LearningRateScheduler(self.scheduler)
        cbks = [change_lr, tb_cb]

        # buid model firstly
        self.build_model(input_shape, output_shape)
        print(self.model.summary())

        methods = {
            'batch_method': {
                'batch_size': self.batch_size,
            },
            'fixed_iter': {
                'steps_per_epoch': self.iterations,
            },
        }

        if method not in methods.keys():
            raise ValueError(
                "method must be `batch_method` or `fixed_iter`")

        self.model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            callbacks=cbks,
            validation_split=validation_rate,
            **methods.get(method),
        )

        if save:
            self.model.save('model.h5')

    @staticmethod
    def scheduler(epoch):
        if epoch <= 30:
            return 0.05
        if epoch <= 60:
            return 0.02
        return 0.01