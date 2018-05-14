import os
import numpy as np


def load_grid(path, number):
    # If path is not absolute we make it
    if not os.path.isabs(path):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(file_dir, path)

    filename = str(number) + ".npz"

    npzfile = np.load(path+filename)
    features = npzfile["features"]
    labels = npzfile["labels"]

    return (features, labels)
