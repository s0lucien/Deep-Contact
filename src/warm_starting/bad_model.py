from .model import Model

class BadModel (Model):
    def __init__(self):
        pass

    def Step(self, world, timeStep, velocityIterations, positionIterations):
        pass

    # Provides a bad prediction irregardless of input
    def Predict(self, contact):
        predictions = []

        m = contact.manifold
        for point in m.points:
            predictions.append((point.id, 10000, 10000))

        return predictions
