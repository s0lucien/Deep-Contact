from model import Model
from util import copyWorld

class CopyWorldModel (Model):
    def __init__(self):
        pass

    # Creates a copy of the world, tells it to take a step
    # and prepares a dictionary with impulse-results for use by Predict
    def Step(self, world, timeStep, velocityIterations, positionIterations,
             velocityThreshold, positionThreshold):
        copy = copyWorld(world)
        copy.enableContinuous = False

        copy.Step(timeStep, velocityIterations, positionIterations, velocityThreshold, positionThreshold)

        self.predictions = {}
        for c in copy.contacts:
            res = []
            for p in c.manifold.points:
                res.append((p.id, p.normalImpulse, p.tangentImpulse))
            self.predictions[(c.fixtureA.body.userData, c.fixtureB.body.userData)] = res

    # Predict a result by looking up in dictionary
    def Predict(self, contact):
        idA = contact.fixtureA.body.userData
        idB = contact.fixtureB.body.userData
        res1 = self.predictions.get((idA, idB), [])
        res2 = self.predictions.get((idB, idA), [])
        return res1+res2
