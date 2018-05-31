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
        # Create the data frames
        df_b = world_body_dataframe(world)
        df_c = world_contact_dataframe(world)
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

        # Tell the gridmanager to add the required grids
        self.gm.add_grids(grid_c, ["ni", "ti"])
    
        # Tell the gridmanager to add the required interpolation functions
        self.gm.create_interp(["ni", "ti"])
        
    def Predict(self, contact):
        return super(CnnIdentityGridModel, self).Predict(contact)
