from __future__ import absolute_import, division, print_function

from optparse import OptionParser
import tensorflow as tf
import numpy as np

from visual_deep_learning.visual_encoder import visual_encoder, state_decoder
from visual_deep_learning.learning_model import dynamic_predictor


def train(FLAGS):
    # Architecture Definition
    # Frame
    F = tf.placeholder(tf.float32,
                       [None, 6, FLAGS.height, FLAGS.weight, FLAGS.col_dim],
                       name="F")
    F1, F2, F3, F4, F5, F6 = tf.unstack(F, 6, 1)

    # Future Data
    label = tf.placeholder(tf.float32,
                           [None, 8, FLAGS.No, 4],
                           name="label")
    label_part = tf.unstack(label, 8, 1)

    # State Code
    S_label = tf.placeholder(tf.float32,
                             [None, 4, FLAGS.No, 4],
                             name="S_label")
    S_label_part = tf.unstack(S_label, 4, 1)

    # discount factor
    df = tf.placeholder(tf.float32, [], name="DiscountFactor")

    # x and y coordinate channels
    x_cor = tf.placeholder(tf.float32,
                           [None, FLAGS.height, FLAGS.weight, 1],
                           name="x_cor")
    y_cor = tf.placeholder(tf.float32,
                           [None, FLAGS.height, FLAGS.weight, 1],
                           name="y_cor")

    # Visual Encoder
    S1, S2, S3, S4 = visual_encoder(
        F1, F2, F3, F4, F5, F6, x_cor, y_cor, FLAGS)

    # Rolling Dynamic Predictor
    roll_num = FLAGS.roll_num
    roll_in = tf.identity(tf.Variable(tf.zeros([roll_num]),
                                      dtype=tf.float32))
    roll_out = tf.scan(rollout_DP,
                       roll_in,
                       initializer=tf.stack([S1, S2, S3, S4], 0))
    S_pred = tf.unstack(roll_out, 4, 1)[-1]
    S_pred = tf.reshape(tf.stack(tf.unstack(S_pred, FLAGS.batch_num, 1),
                                 0),
                        [-1, FLAGS.No, FLAGS.Ds])

    # State Decoder
    S = tf.concat([S1, S2, S3, S4, S_pred], 0)
    out_sd = state_decoder(S, FLAGS)
    S_est = np.zeros(4, dtype=object)
    for i in range(4):
        S_est[i] = tf.slice(out_sd,
                            [FLAGS.batch_num*i, 0, 0],
                            [FLAGS.batch_num, -1, -1])
    label_pred = tf.reshape(tf.slice(out_sd,
                                     [FLAGS.batch_num*4, 0, 0],
                                     [-1, -1, -1]),
                            [FLAGS.batch_num, roll_num, FLAGS.No, 4])
    label_pred8 = tf.unstack(label_pred, roll_num, 1)[:8]

    # loss and optimizer
    mse = df * tf.reduce_mean(
        tf.reduce_mean(
            tf.square(label_pred8[0] - label_part[0]), [1, 2])
    )
    for i in range(1, 8):
        mse += (df ** (i+1)) * tf.reduce_mean(
            tf.reduce_mean(tf.square(label_pred8[i] - label_part[i]), [1, 2]))
    mse = mse/8

    ve_loss = tf.reduce_mean(
        tf.reduce_mean(
            tf.square(S_est[0] - S_label_part[0]), [1, 2]))
    for i in range(1, 4):
        ve_loss += tf.reduce_mean(
            tf.reduce_mean(
                tf.square(S_est[i]-S_label_part[i]), [1, 2])
        )
    ve_loss = ve_loss/4

    total_loss = mse + ve_loss
    optimizer = tf.train.AdamOptimizer(0.0005)
    trainer = optimizer.minimize(total_loss)

    # tensorboard
    tf.summary.scalar('tr_loss', total_loss)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.log_dir)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()


def rollout_DP(prev_out):
    # rolling DP to roll_num
    S1, S2, S3, S4 = tf.unstack(prev_out, 4, 0)
    S_pred = dynamic_predictor(S1, S2, S3, S4, FLAGS)
    res = tf.stack([S2, S3, S4, S_pred], 0)

    return res


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
