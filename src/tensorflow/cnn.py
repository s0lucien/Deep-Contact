import math
import tensorflow as tf
import numpy as np

from .cnn_model_fn import cnn_model_fn

class CNN():
    def __init__(self, _):
        params = {}

        # Grid parameters
        params["p_ll"] = (0, 0)
        params["p_ur"] = (30, 30)
        params["xRes"] = 0.5
        params["yRes"] = 0.5
        params["h"]    = 1
        params["N_x"]  = round((params["p_ur"][0]-params["p_ll"][0] + params["xRes"]) / params["xRes"])
        params["N_y"]  = round((params["p_ur"][1]-params["p_ll"][1] + params["yRes"]) / params["yRes"])
        N = params["N_x"] * params["N_x"]

        # Training parameters
        params["eta"]   = 0.01
        params["lmbda"] = 0.1   # for batch normalization


        # Convolution layers
        params["convolution_layers"]  = 3
        params["filters"]             = [64, 128, 256]
        params["kernel_size"]         = [16, 8, 4]
        params["convolution_padding"] = "same"

        params["max_pool"]     = [True for _ in range(params["convolution_layers"])]
        params["pool_size"]    = [2 for _ in range(params["convolution_layers"])]
        params["pool_stride"]  = [2 for _ in range(params["convolution_layers"])]
        params["pool_padding"] = "same"

        params["batch_normalization"] = [True for _ in range(params["convolution_layers"])]


        # Dense layers
        params["dense_layers"] = 1
        params["dense_units"] = [2 * N]

        params["dense_dropout"] = [True for _ in range(params["dense_layers"])]
        params["dense_dropout_rate"] = [0.4]


        # Output layer
        params["output_dropout"] = True      # Controls whether to do dropout before output layer
        params["output_dropout_rate"] = 0.4

        params["output_units"] = 2 * N
        params["output_grids"] = 2


        # Path to model storage
        params["model_path"] = "./src/tensorflow/models/cnn/"

        # Attributes that the model uses
        params["body_channels"] = ["mass", "vx", "vy", "omega"]
        params["contact_channels"]= ["nx", "ny"]
        params["N_c"] = len(params["body_channels"]) + len(params["contact_channels"])


        self.params = params
        self.estimator = tf.estimator.Estimator(
            model_fn=cnn_model_fn,
            model_dir=params["model_path"],
            params=params
        )


    def train(self, features, labels, train_params):
        # We create the input function
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"x": features},
            y = labels,
            batch_size = train_params["batch_size"],
            num_epochs = train_params["num_epochs"],
            shuffle = True
        )

        # We train
        self.estimator.train(
            input_fn=train_input_fn
        )


    def evaluate(self, features, labels):
        # We create the input function
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"x": features},
            y = labels,
            num_epochs = 1,
            shuffle = False
        )

        # We do the evaluation
        eval_results = self.estimator.evaluate(input_fn=eval_input_fn)

        return eval_results


    def predict(self, features):
        # We create the input function
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"x": features},
            num_epochs = 1,
            shuffle = False
        )

        # We do the predictions
        pred_results = self.estimator.predict(input_fn=pred_input_fn)

        # For some reason the result is some unsubscriptable iterator object
        # There should only be a single pred_dict
        for pred_dict in pred_results:
            ni = pred_dict["ni"]
            ti = pred_dict["ti"]

        return (ni, ti)
