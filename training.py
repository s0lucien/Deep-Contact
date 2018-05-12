from __future__ import absolute_import, division, print_function

from optparse import OptionParser
import tensorflow as tf
import numpy as np
import cv2

from visual_deep_learning.visual_encoder import visual_encoder, state_decoder
from visual_deep_learning.learning_model import dynamic_predictor


def sess_train(FLAGS):
    # Architecture Definition
    # Frame
    F = tf.placeholder(tf.float32,
                       [None, 6, FLAGS.hight, FLAGS.wight, FLAGS.col_dim],
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
                           [None, FLAGS.hight, FLAGS.wight, 1],
                           name="x_cor")
    y_cor = tf.placeholder(tf.float32,
                           [None, FLAGS.hight, FLAGS.wight, 1],
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

    return sess, trainer, merged, writer


def rollout_DP(prev_out):
    # rolling DP to roll_num
    S1, S2, S3, S4 = tf.unstack(prev_out, 4, 0)
    S_pred = dynamic_predictor(S1, S2, S3, S4, FLAGS)
    res = tf.stack([S2, S3, S4, S_pred], 0)

    return res


def read_train_images(path, FLAGS, tr_ratio=1, val_ratio=0):
    '''
    parameters:
        path: traing image path
        tr_ratio = percentage of training images
        val_ratio = percentage of validation images

    return:
        tr_data, tr_label, tr_S_label
    '''
    # Get Training Image and Data
    frame_num = FLAGS.frame_num
    set_num = FLAGS.set_num
    hight = FLAGS.hight
    wight = FLAGS.wight
    col_dim = FLAGS.col_dim

    total_img = np.zeros(
        (set_num, frame_num, hight, wight, col_dim),
        dtype=float,
    )
    for i in range(set_num):
        for j in range(frame_num):
            total_img[i, j] = cv2.imread(
                path + 'train_' + str(i) + '/' + str(j) + '.png',
            )[:, :, :col_dim]

    total_data = np.zeros(
        (set_num, frame_num, FLAGS.No*5), dtype=float)
    for i in range(set_num):
        # read xml file
        f = open()
        total_data[i] = None

    total_data = np.reshape(
        total_data, [set_num, frame_num, FLAGS.No, 5])

    # reshape img and data
    input_img = np.zeros(
        (
            set_num*(frame_num-14+1),
            6,
            FLAGS.hight,
            FLAGS.wight,
            FLAGS.col_dim,
        ),
        dtype=float,
    )

    output_label = np.zeros(
        (
            FLAGS.set_num*(frame_num-14+1),
            8,
            FLAGS.No,
            4,
        ),
        dtype=float,
    )
    output_S_label = np.zeros(
        (
            FLAGS.set_num*(frame_num-14+1),
            4,
            FLAGS.No,
            4,
        ),
        dtype=float,
    )

    for i in range(set_num):
        for j in range(frame_num-14+1):
            input_img[i*(frame_num-14+1)+j] = total_img[i, j:j+6]
            output_label[i*(frame_num-14+1)+j] = np.reshape(
                total_data[i, j+6:j+14],
                [8, FLAGS.No, 5],
            )[:, :, 1:5]
            output_S_label[i*(frame_num-14+1)+j] = np.reshape(
                total_data[i, j+2:j+6],
                [4, FLAGS.No, 5],
            )[:, :, 1:5]

    # shuffle
    tr_data_num = int(len(input_img)*tr_ratio)
    val_data_num = int(len(input_img)*val_ratio)
    total_idx = list(range(len(input_img)))
    np.random.shuffle(total_idx)
    mixed_img = input_img[total_idx]
    mixed_label = output_label[total_idx]
    mixed_S_label = output_S_label[total_idx]

    tr_data = mixed_img[:tr_data_num]
    tr_label = mixed_label[:tr_data_num]
    tr_S_label = mixed_S_label[:tr_data_num]

    val_data = mixed_img[tr_data_num:(tr_data_num+val_data_num)]
    val_label = mixed_label[tr_data_num:(tr_data_num+val_data_num)]
    val_S_label = mixed_S_label[tr_data_num:(tr_data_num+val_data_num)]

    return tr_data, tr_label, tr_S_label


def xy_cor(FLAGS):
    # x-cor and y-cor setting
    nx, ny = (FLAGS.wight, FLAGS.hight)

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)

    xv, yv = np.meshgrid(x, y)
    xv = np.reshape(xv, [FLAGS.hight, FLAGS.wight, 1])
    yv = np.reshape(yv, [FLAGS.hight, FLAGS.wight, 1])

    xcor = np.zeros(
        (FLAGS.batch_num*5, FLAGS.hight, FLAGS.wight, 1),
        dtype=float,
    )
    ycor = np.zeros(
        (FLAGS.batch_num*5, FLAGS.hight, FLAGS.wight, 1),
        dtype=float,
    )
    for i in range(FLAGS.batch_num*5):
        xcor[i] = xv
        ycor[i] = yv

    return xcor, ycor


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
    parser.add_option(
        '--wight', type=int, help='wight',
    )
    parser.add_option(
        '--hight', type=int, help='hight',
    )

    FLAGS, args = parser.parse_args()
    FLAGS.No = 2
    print(xy_cor(FLAGS))

    print(FLAGS.No)
