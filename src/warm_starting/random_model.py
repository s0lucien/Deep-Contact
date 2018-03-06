from model import Model

import random

class RandomModel (Model):
    # We manually choose a seed to ensure the same 'random' numbers each time
    def __init__(self, seed):
        random.seed(seed)
        pass

    def Step(self, world, timeStep, velocityIterations, positionIterations,
             velocityThreshold, positionThreshold):
        pass

    # Predicts a random, some-what 'reasonable' set of impulses
    def Predict(self, contact):
        predictions = []

        m = contact.manifold
        for point in m.points:
            # Normal impulses seems to be in the range 0 to 20
            normal = random.uniform(0, 20)

            # Tangential impulses seems to be in the range -10 to 10
            tangent = random.uniform(-10, 10)

            predictions.append((point.id, normal, tangent))

        return predictions
