from .model import Model

class IdentityGridModel ():
    # Initializes the Model
    def __init__(self, grid_parameters):
        # initialize grid
        self.grid = None

    # Tells the model to take a step
    def Step(self, world, timeStep, velocityIterations, positionIterations):
        for c in world.contacts:
            # for circles there should only be 1 point
            for p in c.manifold.points:
                # Add impulses to grid
                p.normalImpulse
                p.tangentImpulse
                pass


    # Takes a contact and returns a tuple of id, normal impulse and tangential impulse for each manifold point
    def Predict(self, contact):
        for p om c.manifold.points:
            # Get point warm start impulses from grid
            p.id
            pass
        return []
