import time
import numpy as np
import tensorflow as tf

from ..gen_data.load_grid import load_grid
from .cnn import CNN

# Disables stupid tensorflow warnings about cpu instructions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Training data
# Path is relative to the gen_data directory
train_path = "data/grid3030100_05_1/"
train_numbers = range(1, 11)
train_steps = 600

# Training parameters
train_params = {}
train_params["batch_size"] = 50
train_params["num_epochs"] = 3

# Evaluation data
# Path is relative to the gen_data directory
eval_path = train_path
eval_numbers = [0]
eval_steps = 600

# Decides whether to train
train = True
# Decides whether to evaluate
evaluate = False


# Create the cnn
cnn = CNN({})

# Training
if train:
    for i in range(len(train_numbers)):
        n = train_numbers[i]
        print("Training on dataset %d of %d" % (i+1, len(train_numbers)))

        # We load the training data
        start = time.time()

        train_features, train_labels = load_grid(train_path, n)

        print("Loading training data took: %d s" % (time.time() - start))


        # We train the model
        start = time.time()

        cnn.train(train_features, train_labels, train_params)

        print("Training took: %d s" % (time.time()-start))


# Evaluation
if evaluate:
    for i in range(len(eval_numbers)):
        n = eval_numbers[i]
        print("Evaluating dataset %d of %d" % (i+1, len(eval_numbers)))

        # We load the evaluation data
        start = time.time()

        eval_features, eval_labels = load_grid(eval_path, n)

        print("Loading evaluation data took: %d s" % (time.time() - start))

        # Evaluate the model and print results
        start = time.time()

        eval_results = cnn.evaluate(eval_features, eval_labels)
        print(eval_results)

        print("Evaluation took: %d s" % (time.time()-start))
