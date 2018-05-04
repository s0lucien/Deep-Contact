from .model import Model
from ..sph.gridsplat import SPHGridManager, world_body_dataframe, world_contact_dataframe

class IdentityGridModel (Model):
    def __init__(self, p_ll, p_ur, xRes, yRes, h):
        # Initialize the grid
        self.gm = SPHGridManager(p_ll, p_ur, xRes, yRes, h)


    def Step(self, world, timeStep, velocityIterations, positionIterations):
        # Create the data frames
        df_b = world_body_dataframe(world)
        df_c = world_contact_dataframe(world)

        # Tell the gridmanager to create the required grids
        self.gm.create_grids(df_b, df_c, channels=["ni", "ti"])


    def Predict(self, contact):
        predictions = []

        for i in range(contact.manifold.pointCount):
            px = contact.worldManifold.points[i][0]
            py = contact.worldManifold.points[i][1]

            id = contact.manifold.points[i].id
            normalImpulse = self.gm.query(px, py, "ni")
            tangentImpulse = self.gm.query(px, py, "ti")

            predictions.append((id, normalImpulse, tangentImpulse))
        return predictions
