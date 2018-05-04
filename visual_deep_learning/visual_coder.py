# This file includes the model for visual encoder 
# and visual decoder

from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import (
    Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D,
)
from keras.layers.normalization import BatchNormalization
import keras.models as models
from keras.layers.core import (
    Layer, Dense, Dropout, Activation, Flatten, Reshape,
)
from keras.utils import np_utils
from keras import backend as K
import tensorflow as T

import os

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = \
	'mode=FAST_RUN, floatX=float32, optimizer=fast_compile'


class UnPooling2D(Layer):
    """
	A 2D Repeat layer
	Example:
		| x y |   -->  | x x y y|
		| z a |        | z z a a|
	"""
    def __init__(self, poolsize=(2, 2)):
        super(UpSampling2D, self).__init__()
        self.input = T.tensor4()
        self.poolsize = poolsize

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1],
                self.poolsize[0] * input_shape[2],
                self.poolsize[1] * input_shape[3])

    def get_output(self, train):
        X = self.get_input(train)
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        output = X.repeat(s1, axis=2).repeat(s2, axis=3)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "poolsize":self.poolsize}


class AutoEncoder(object):
	def __init__(self):
		self.model = models.Sequential()

	def build_model(self, input_shape):
		self.model.add(Layer(input_shape=input_shape))
		self.model.encoding_layers = self.create_encoding_layers()
		self.model.decoding_layers = self.create_decoding_layers()

		for layer in self.model.encoding_layers:
			self.model.add(layer)

		for layer in self.model.decoding_layers:
			self.model.add(layer)

	@staticmethod
	def create_encoding_layers():
		kernel = 3
		pad = 1
		pool_size = 2

		return [
			ZeroPadding2D(padding=(pad,pad)),
			Convolution2D(64, kernel, kernel, border_mode='valid'),
			BatchNormalization(),
			Activation('relu'),
			MaxPooling2D(pool_size=(pool_size, pool_size)),

			ZeroPadding2D(padding=(pad,pad)),
			Convolution2D(128, kernel, kernel, border_mode='valid'),
			BatchNormalization(),
			Activation('relu'),
			MaxPooling2D(pool_size=(pool_size, pool_size)),

			ZeroPadding2D(padding=(pad,pad)),
			Convolution2D(256, kernel, kernel, border_mode='valid'),
			BatchNormalization(),
			Activation('relu'),
			MaxPooling2D(pool_size=(pool_size, pool_size)),

			ZeroPadding2D(padding=(pad,pad)),
			Convolution2D(512, kernel, kernel, border_mode='valid'),
			BatchNormalization(),
			Activation('relu')
		]

	@staticmethod
	def create_decoding_layers():
		kernel = 3
		pad = 1
		pool_size = 2

		return [
			ZeroPadding2D(padding=(pad,pad)),
			Convolution2D(512, kernel, kernel, border_mode='same'),
			BatchNormalization(),

			UpSampling2D(size=(pool_size,pool_size)),
			ZeroPadding2D(padding=(pad,pad)),
			Convolution2D(256, kernel, kernel, border_mode='same'),
			BatchNormalization(),

			UpSampling2D(size=(pool_size,pool_size)),
			ZeroPadding2D(padding=(pad,pad)),
			Convolution2D(128, kernel, kernel, border_mode='same'),
			BatchNormalization(),

			UpSampling2D(size=(pool_size,pool_size)),
			ZeroPadding2D(padding=(pad,pad)),
			Convolution2D(64, kernel, kernel, border_mode='same'),
			BatchNormalization(),
		]

#a = AutoEncoder()
#a.build_model(input_shape=(1, 50, 50))
#print(a.model.summary)
