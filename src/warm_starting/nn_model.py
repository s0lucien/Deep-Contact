import numpy as np

from .model import Model
from ..sph.gridsplat import world_body_dataframe, world_contact_dataframe, SPHGridManager

# Disables stupid tensorflow warnings about cpu instructions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class NNModel ():
    def __init__(self, cnn):
        self.cnn = cnn


    def Step(self, world, timeStep, velocityIterations, positionIterations):
        self.predictions = {}

        # We create the dataframes for the current world
        b_df = world_body_dataframe(world)
        c_df = world_contact_dataframe(world)

        # No reason to waste time on calling model if there are no contacts
        if c_df.empty:
            return

        # We create the grids for the current world
        params = self.cnn.params
        self.gm = SPHGridManager(params["p_ll"], params["p_ur"],
                                 params["xRes"], params["yRes"],
                                 params["h"])
        self.gm.create_grids(b_df, params["body_channels"])
        self.gm.create_grids(c_df, params["contact_channels"])

        # We transform the grids into input for the cnn
        grids = []
        for grid in self.gm.grids.values():
            grids.append(grid)

        grids = np.array(grids, dtype=np.float32)
        grids = np.rollaxis(grids, 0, 3)

        # We feed the input to the cnn
        ni_grid, ti_grid = self.cnn.predict(grids)
        ni_grid = np.reshape(ni_grid, (params["N_x"], params["N_y"]))
        ti_grid = np.reshape(ti_grid, (params["N_x"], params["N_y"]))

        # We assign ids to contacts and create a list of contact positions
        ic = 0
        points = []
        for c in world.contacts:
            if c.touching:
                c.userData = ic
                ic += 1

                #for i in range(contact.manifold.pointCount):   # We assume circles!!!
                px = c.worldManifold.points[0][0]
                py = c.worldManifold.points[0][1]
                points.append([px, py])

        # Tell GM to transfer from grids to contacts
        grids = np.array([ni_grid, ti_grid])
        points = np.array(points)
        values = self.gm.grids_to_particles(grids, points)

        # We store the results for easy acces
        for i in range(len(values)):
            self.predictions[i] = values[i]


    def Predict(self, contact):
        # Again, we assume circles for now
        ic = contact.userData
        prediction = self.predictions.get(ic, [])
        if len(prediction) == 0:
            print("Contact not found!")
            return [(contact.manifold.points[0].id, 0, 0)]

        normalImpulse  = prediction[0]
        tangentImpulse = prediction[1]

        return [(contact.manifold.points[0].id, normalImpulse, tangentImpulse)]
