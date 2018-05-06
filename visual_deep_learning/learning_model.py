from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np


def interaction_net(S, FLAGS, idx):
    fil_num = 64
    M = tf.unstack(S, FLAGS.No, 1)

    # Self-Dynamics MLP
    SD_in = tf.reshape(S, [-1, FLAGS.Ds])
    with tf.variable_scope('self-dynamics' + str(idx)):
        w1 = tf.get_variable('w1', shape=[FLAGS.Ds, fil_num])
        b1 = tf.get_variable('b1', shape=[fil_num])
        h1 = tf.nn.relu(tf.matmul(SD_in, w1) + b1)
        w2 = tf.get_variable('w2', shape=[fil_num, fil_num])
        b2 = tf.get_variable('b2', shape=[fil_num])
        h2 = tf.matmul(h1, w2) + b2+h1
        M_self = tf.reshape(h2, [-1, FLAGS.No, fil_num])

    # Relation MLP
    rel_num = int((FLAGS.No) * (FLAGS.No + 1) / 2)
    rel_in = np.zeros(rel_num, dtype=object)
    for i in range(rel_num):
        row_idx = int(i / (FLAGS.No - 1))
        col_idx = int(i % (FLAGS.No - 1))
        rel_in[i] = tf.concat([M[row_idx], M[col_idx]], 1)
    rel_in = tf.concat(list(rel_in), 0)

    with tf.variable_scope('Relation' + str(idx)):
        w1 = tf.get_variable('w1', shape=[FLAGS.Ds*2, fil_num])
        b1 = tf.get_variable('b1', shape=[fil_num])
        h1 = tf.nn.relu(tf.matmul(rel_in, w1) + b1)

        w2 = tf.get_variable('w2', shape=[fil_num, fil_num])
        b2 = tf.get_variable('b2', shape=[fil_num])
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2 + h1)

        w3 = tf.get_variable('w3', shape=[fil_num, fil_num])
        b3 = tf.get_variable('b3', shape=[fil_num])
        h3 = tf.matmul(h2, w3) + b3+h2

    M_rel = np.zeros(rel_num, dtype=object)

    for i in range(rel_num):
        M_rel[i] = tf.slice(
            h3, [(FLAGS.batch_num)*i, 0], [(FLAGS.batch_num), -1])
    M_rel2 = np.zeros(FLAGS.No, dtype=object)

    for i in range(FLAGS.No):
        for j in range(FLAGS.No - 1):
            M_rel2[i] += M_rel[i * (FLAGS.No - 1) + j]
    M_rel2 = tf.stack(list(M_rel2), 1)

    # M_update
    M_update = M_self + M_rel2

    # Affector MLP
    aff_in = tf.reshape(M_update, [-1, fil_num])
    with tf.variable_scope('Affector' + str(idx)):
        w1 = tf.get_variable('w1', shape=[fil_num, fil_num])
        b1 = tf.get_variable('b1', shape=[fil_num])
        h1 = tf.nn.relu(tf.matmul(aff_in, w1) + b1 + aff_in)

        w2 = tf.get_variable('w2', shape=[fil_num, fil_num])
        b2 = tf.get_variable('b2', shape=[fil_num])
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2 + h1)

        w3 = tf.get_variable('w3', shape=[fil_num, fil_num])
        b3 = tf.get_variable('b3', shape=[fil_num])
        h3 = tf.matmul(h2, w3) + b3 + h2

    M_affect = tf.reshape(h3, [-1, FLAGS.No, fil_num])

    # Output MLP
    M_i_M_affect = tf.concat([S, M_affect], 2)
    out_in = tf.reshape(M_i_M_affect, [-1, FLAGS.Ds+fil_num])

    with tf.variable_scope('Output' + str(idx)):
        w1 = tf.get_variable('w1', shape=[FLAGS.Ds+fil_num, fil_num])
        b1 = tf.get_variable('b1', shape=[fil_num])
        h1 = tf.nn.relu(tf.matmul(out_in, w1) + b1)

        w2 = tf.get_variable('w2', shape=[fil_num, FLAGS.Ds])
        b2 = tf.get_variable('b2', shape=[FLAGS.Ds])
        h2 = tf.matmul(h1, w2) + b2

        h2_out = tf.reshape(h2, [-1, FLAGS.No, FLAGS.Ds])

    return h2_out


# Just one learning model here, maybe more,
# like LSTM, RNN
