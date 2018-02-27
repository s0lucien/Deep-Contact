

class Model ():
    def __init__(self):
        pass

    # Tells the model to take a step
    def Step(self, world, timeStep, velocityIterations, positionIterations):
        pass

    # Takes a contact and returns a tuple of id, normal impulse and tangential impulse for each manifold point
    def Predict(self, contact):
        return []
