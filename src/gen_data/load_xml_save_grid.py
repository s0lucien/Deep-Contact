import os
import time
import numpy as np
import xml.etree.ElementTree as ET

from .load_xml_return_grid import load_xml_return_grid


# Grid parameters
# Grid lower left point
p_ll = (0, 0)
# Grid upper right point
p_ur = (50, 50)
# Grid x-resolution
xRes = 0.75
# Grid y-resolution
yRes = 0.75
# Support radius
h = 1

# The body attributes we will use
body_channels = ["mass", "inertia", "vx", "vy", "omega"]
# The contact attributes we will use
contact_channels = ["ni", "ti", "nx", "ny"]
# The body and contact attributes to use as input
feature_channels = ["mass", "inertia", "vx", "vy", "omega", "nx", "ny"]
# The body and contact attributes to use as label
label_channels = ["ni", "ti"]

# Data information
xml_path  = "../gen_data/data/xml/"
grid_path = "../gen_data/data/grid/"
numbers = [25]
steps = 1000


for i in range(len(numbers)):
    print("Processing dataset %d of %d" % (i+1, len(numbers)))
    start = time.time()
    n = numbers[i]

    # Load the xml file and convert to grids
    features, labels = load_xml_return_grid(
        xml_path, n, steps,
        body_channels, contact_channels,
        feature_channels, label_channels,
        p_ll, p_ur, xRes, yRes, h
    )

    # If path is not absolute we make it
    if not os.path.isabs(grid_path):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        grid_path = os.path.join(file_dir, grid_path)

    # Save the grids using numpy
    file = grid_path + str(n)
    np.savez(file, features=features, labels=labels)
    print("Processing took %d s" % (time.time() - start))
