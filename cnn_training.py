from __future__ import absolute_import, division, print_function

import numpy as np

from visual_deep_learning.cnn_model import learing_model


def load_grid(path):
	pass


if __name__ == '__main__':
	from optparse import OptionParser

	parser = OptionParser()
	parser.add_option('-p', '--path', dest='path')

	options, _ = parser.parse_args()

	model = learing_model(
		log_dir='./log/',
	)

	x_tr, y_tr = load_grid(options.path)

	model.train(x_tr, y_tr)

