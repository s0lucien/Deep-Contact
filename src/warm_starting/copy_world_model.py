from model import Model
from util import copyWorld

class CopyWorldModel (Model):
    # Creates a copy of the given world by creating copies of all bodies
    def __init__(self):
        pass

    # Creates a copy of the world, tells it to take a step
    # and prepares a dictionary with impulse-results for use by Predict
    def Step(self, world, timeStep, velocityIterations, positionIterations):
        copy = copyWorld(world)
        copy.Step(timeStep, velocityIterations, positionIterations)

        self.predictions = {}
        for c in copy.contacts:
            if c.touching:
                res = []
                for p in c.manifold.points:
                    res.append((p.id, p.normalImpulse, p.tangentImpulse))
                self.predictions[(c.fixtureA.body.userData, c.fixtureB.body.userData)] = res

    # Predict a result by looking up in dictionary
    def Predict(self, contact):
        return self.predictions.get((contact.fixtureA.body.userData, contact.fixtureB.body.userData), [])
