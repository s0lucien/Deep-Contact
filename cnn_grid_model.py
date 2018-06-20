import numpy as np


from src.warm_starting.identity_grid_model import IdentityGridModel
from src.sph.gridsplat import SPHGridManager, world_body_dataframe, world_contact_dataframe


class CnnIdentityGridModel(IdentityGridModel):
    """docstring for CnnIdentityGridModel"""
    def __init__(self, p_ll, p_ur, xRes, yRes, h, learning_model):
        super(CnnIdentityGridModel, self).__init__(
            p_ll, p_ur, xRes, yRes, h
        )
        self.learning_model = learning_model

    def Step(self, world, timeStep, velocityIterations, positionIterations):
        self.predictions = {}

        # Create the data frames
        df_b = world_body_dataframe(world)
        df_c = world_contact_dataframe(world)

        if df_c.empty:
            return

        self.gm.create_grids(
            df_b, ["mass", "vx", "vy", "omega"]
        )
        self.gm.create_grids(df_c, ["nx"])
        train_grid = np.stack(
            [
                self.gm.grids.get(c)
                for c in ["mass", "vx", "vy", "omega", "nx"]
            ]
        )
        grid_c = self.learning_model.predict(train_grid.reshape((1, 41, 41, 5))).reshape((2, 41, 41))

        # do the prediction
        points = []
        for ic, c in enumerate(world.contacts):
            if c.touching:
                c.userData = ic

                px = c.worldManifold.points[0][0]
                py = c.worldManifold.points[0][1]

                points.append([px, py])

        points = np.array(points)
        values = self.gm.grids_to_particles(grid_c, points)
        for i in range(len(values)):
            self.predictions[i] = values[i]
        
    def Predict(self, contact):
        ic = contact.userData
        prediction = self.predictions.get(ic, [])
        if len(prediction) == 0:
            return [(contact.manifold.points[0].id, 0, 0)]

        normalImpulse  = prediction[0]
        tangentImpulse = prediction[1]

        return [(contact.manifold.points[0].id, normalImpulse, tangentImpulse)]
