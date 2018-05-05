# using tensorflow stuff
from __future__ import division

import tensorflow as tf
import numpy as np


def conv_variable(weight_shape):
    w = weight_shape[0]
    h = weight_shape[1]
    input_channels  = weight_shape[2]
    output_channels = weight_shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]
    weight = tf.Variable(
        tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias = tf.Variable(
        tf.random_uniform(bias_shape, minval=-d, maxval=d))

    return weight, bias


def conv2d(x, W, stride):
  return tf.nn.conv2d(
      x, W, strides=[1, stride, stride, 1], padding="SAME")


def maxpool2d(x, k=2):
  return tf.nn.max_pool(
      x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def visual_encoder(F1, F2, F3, F4, F5, F6, x_cor, y_cor, FLAGS):
    '''
    parameters:
        Fn: Consecutive input images
        x_cor: x-coordinate
        y_cor: y-coordinate
        FLAGS: Learning settings

    return:
        state_code(1, 2, 3, 4)
    '''
    F1F2=tf.concat([F1,F2],3)
    F2F3=tf.concat([F2,F3],3)
    F3F4=tf.concat([F3,F4],3)
    F4F5=tf.concat([F4,F5],3)
    F5F6=tf.concat([F5,F6],3)

    # TODO: a func for getting input pairs encode
    pair1, pair2, pair3, pair4, pair5 = func(
        F1F2, F2F3, F3F4, F4F5, F5F6, x_cor, y_cor, FLAGS)

    concated_pair = tf.concat([pair1, pair2, pair3, pair4, pair5], 0)

    # shared a linear layer
    fil_num = FLAGS.fil_num
    w0 = tf.Variable(
        tf.truncated_normal([fil_num, FLAGS.No*FLAGS.Ds], stddev=0.1), dtype=tf.float32)
    b0 = tf.Variable(
        tf.zeros([FLAGS.No*FLAGS.Ds]), dtype=tf.float32)
    h0 = tf.matmul(concated_pair, w0) + b0

    enpair1 = tf.reshape(
        tf.slice(h0, [0, 0], [FLAGS.batch_num, -1]),
        [-1, FLAGS.No, FLAGS.Ds],
    )
    enpair2 = tf.reshape(
        tf.slice(h0, [FLAGS.batch_num, 0], [FLAGS.batch_num, -1]),
        [-1, FLAGS.No, FLAGS.Ds],
    )
    enpair3 = tf.reshape(
        tf.slice(h0, [FLAGS.batch_num * 2, 0], [FLAGS.batch_num, -1]),
        [-1,FLAGS.No,FLAGS.Ds],
    )
    enpair4 = tf.reshape(
        tf.slice(h0,[FLAGS.batch_num * 3, 0], [FLAGS.batch_num, -1]),
        [-1,FLAGS.No,FLAGS.Ds],
    )
    enpair5 = tf.reshape(
        tf.slice(h0, [FLAGS.batch_num * 4, 0], [FLAGS.batch_num, -1]),
        [-1,FLAGS.No,FLAGS.Ds],
    )

    three1 = tf.concat([enpair1, enpair2], 2)
    three2 = tf.concat([enpair2, enpair3], 2)
    three3 = tf.concat([enpair3, enpair4], 2)
    three4 = tf.concat([enpair4, enpair5], 2)

    # shared MLP
    three=tf.concat([three1, three2, three3, three4], 0)
    three=tf.reshape(three, [-1, FLAGS.Ds * 2])

    w1 = tf.Variable(
        tf.truncated_normal([FLAGS.Ds * 2, 64], stddev=0.1), dtype=tf.float32)
    b1 = tf.Variable(
        tf.zeros([64]), dtype=tf.float32)
    h1 = tf.nn.relu(
        tf.matmul(three, w1) + b1)
    w2 = tf.Variable(
        tf.truncated_normal([64, 64], stddev=0.1), dtype=tf.float32)
    b2 = tf.Variable(
        tf.zeros([64]), dtype=tf.float32)
    h2 = tf.nn.relu(
        tf.matmul(h1, w2) + b2 + h1)
    w3 = tf.Variable(
        tf.truncated_normal([64, FLAGS.Ds], stddev=0.1), dtype=tf.float32)
    b3 = tf.Variable(
        tf.zeros([FLAGS.Ds]), dtype=tf.float32)
    h3 = tf.matmul(h2, w3) + b3 + h2
    h3 = tf.reshape(h3, [-1, FLAGS.No, FLAGS.Ds])

    S1 = tf.slice(h3, [0, 0, 0],[FLAGS.batch_num, -1, -1])
    S2 = tf.slice(h3, [FLAGS.batch_num, 0, 0], [FLAGS.batch_num, -1, -1])
    S3 = tf.slice(h3, [FLAGS.batch_num*2, 0, 0], [FLAGS.batch_num, -1, -1])
    S4 = tf.slice(h3, [FLAGS.batch_num*3, 0, 0], [FLAGS.batch_num, -1, -1])

    return S1, S2, S3, S4
