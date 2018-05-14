import os
import numpy as np
import xml.etree.ElementTree as ET

from ..sph.gridsplat import SPHGridManager, xml_body_dataframe, xml_contact_dataframe


# Function for loading xml dataset and returning grids
def load_xml_return_grid(path, number, steps,
                         body_channels, contact_channels,
                         feature_channels, label_channels,
                         p_ll, p_ur, xRes, yRes, h):

    # We determine thee various sizes and shapes
    N_x = round((p_ur[0] - p_ll[0]) / xRes)
    N_y = round((p_ur[1] - p_ll[1]) / yRes)

    # If path is not absolute we make it
    if not os.path.isabs(path):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(file_dir, path)
    path += str(number) + "/"

    # We create the grid manager
    gm = SPHGridManager(p_ll, p_ur, xRes, yRes, h)

    features = []
    labels = []
    for i in range(1, steps+1):
        # We load the xml file
        try:
            filename = str(number) + "_" + str(i) + ".xml"
            xml = ET.ElementTree(file=path+filename).getroot()
        except:
            continue

        # We transfer the xml data onto grids
        b_df = xml_body_dataframe(xml)
        c_df = xml_contact_dataframe(xml)

        gm.reset()
        gm.create_grids(b_df, body_channels)
        gm.create_grids(c_df, contact_channels)

        # We separate the input and the label grids
        l = []
        for c in label_channels:
            l.append(gm.grids[c])
        l = np.array(l, dtype=np.float32)

        fs = []
        for c in feature_channels:
            fs.append(gm.grids[c])
        fs = np.array(fs, dtype=np.float32)

        # We reshape our data
        l = np.ndarray.flatten(l)

        fs = np.rollaxis(fs, 0, 3)

        # We add the data
        features.append(fs)
        labels.append(l)

    # We convert the data to a float32 numpy array
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    return (features, labels)
