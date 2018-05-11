from __future__ import absolute_import, division, print_function

from optparse import OptionParser
import tensorflow as tf
import numpy as np


def train():
    pass


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option(
        '--log_dir', type=str, default='./logs/',
        help='Summaries log directry',
    )
    parser.add_option(
        '--batch_num', type=int, default=4, dest='batch_num',
        help='The number of data on each mini batch',
    )
    parser.add_option(
        '--max_epoches', type=int, default=80000,
        help='Maximum limitation of epoches',
    )
    parser.add_option(
        '--Ds', type=int, default=64, dest='Ds',
        help='The State Code Dimension',
    )
    parser.add_option(
        '--fil_num', type=int, default=128, dest='fil_num',
        help='The Number of filters',
    )
    FLAGS, args = parser.parse_args()
    FLAGS.No = 2
    print(FLAGS.No)
