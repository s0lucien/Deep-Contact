from .model import Model
from ..sph.kernel import W_poly6_2D, spiky_2D
from ..sph.gridsplat import SPHGridManager, world_body_dataframe, world_contact_dataframe
from .util import copyWorld

class IdentityGridModel(Model):
    def __init__(self, p_ll, p_ur, xRes, yRes, h, kernel=W_poly6_2D):
        # Initialize the grid
        self.gm = SPHGridManager(p_ll, p_ur, xRes, yRes, h, kernel)

    def Step(self, world, timeStep, velocityIterations, positionIterations):
        copy = copyWorld(world)
        copy.Step(timeStep, velocityIterations, positionIterations)

        # Create the data frames
        df_b = world_body_dataframe(copy)
        df_c = world_contact_dataframe(copy)

        # Tell the gridmanager to create the required grids
        self.gm.create_grids(df_c, ["ni", "ti"])

        # Tell the gridmanager to create the required interpolation functions
        self.gm.create_interp(["ni", "ti"])

    def Predict(self, contact):
        predictions = []

        for i in range(contact.manifold.pointCount):
            px = contact.worldManifold.points[i][0]
            py = contact.worldManifold.points[i][1]

            id = contact.manifold.points[i].id
            normalImpulse = self.gm.query_interp(px, py, "ni")
            tangentImpulse = self.gm.query_interp(px, py, "ti")

            predictions.append((id, normalImpulse, tangentImpulse))

        return predictions
