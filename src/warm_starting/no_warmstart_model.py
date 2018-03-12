from .model import Model

class NoWarmStartModel (Model):
    def __init__(self):
        pass

    def Step(self, world, timeStep, velocityIterations, positionIterations):
        pass

    # Set all starting iterates to 0, effectively disabling warmstart
    def Predict(self, contact):
        predictions = []

        m = contact.manifold
        for point in m.points:
            predictions.append((point.id, 0, 0))

        return predictions
