import numpy as np
import tensorflow as tf


def cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    training = mode == tf.estimator.ModeKeys.TRAIN


    # Input Layer
    data = tf.reshape(features["x"], [-1, params["N_x"], params["N_y"], params["N_c"]])

    for i in range(params["convolution_layers"]):
        # Convolution Layer
        data = tf.layers.conv2d(
            inputs = data,
            filters = params["filters"][i],
            kernel_size = params["kernel_size"][i],
            padding=params["convolution_padding"],
            kernel_initializer = tf.truncated_normal_initializer(),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(params["lmbda"]),
        )

        # Batch normalization
        if params["batch_normalization"][i]:
            data = tf.layers.batch_normalization(
                inputs = data,
                training = training
            )

        # Activation
        data = tf.nn.relu(data)

        # Max Pooling Layer
        if params["max_pool"][i]:
            data = tf.layers.max_pooling2d(
                inputs = data,
                pool_size = params["pool_size"][i],
                strides = params["pool_stride"][i],
                padding = params["pool_padding"]
            )

    # Flatten data for dense layers
    data = tf.layers.flatten(data)

    for i in range(params["dense_layers"]):
        # Dropout
        if params["dense_dropout"][i]:
            data = tf.layers.dropout(
                inputs = data,
                rate = params["dense_dropout_rate"][i],
                training = training
            )

        # Dense Layer
        data = tf.layers.dense(
            inputs = data,
            units = params["dense_units"][i],
            activation = tf.nn.relu
        )

    # Output Layer
    if params["output_dropout"]:
        data = tf.layers.dropout(
            inputs = data,
            rate = params["output_dropout_rate"],
            training = training
        )

    predictions = tf.layers.dense(
        inputs = data,
        units = params["output_units"],
        activation = None
    )


    # Prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        ni, ti = tf.split(predictions, params["output_grids"], axis=1)

        predictions_dict = {
            "ni": ni,
            "ti": ti
        }

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dict)


    # Ignore grid nodes where the label is 0
    flags = tf.to_float(tf.not_equal(labels, 0.0))
    predictions = predictions * flags

    # Calculate Loss (for both TRAIN and EVAL modes)
    average_loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)
    regularization_loss = tf.losses.get_regularization_loss()

    batch_size = tf.shape(labels)[0]
    total_loss = tf.to_float(batch_size) * average_loss

    loss = average_loss + regularization_loss

    # Configure the Training Op (for TRAIN mode)
    if training:
        learning_rate = params["eta"]

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "MAE": tf.metrics.mean_absolute_error(labels=labels, predictions=predictions),
        "RMSE": tf.metrics.root_mean_squared_error(labels=labels, predictions=predictions)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
