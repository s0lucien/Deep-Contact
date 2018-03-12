from .model import Model
from .util import copyWorld

from Box2D import (b2World)
from Box2D import (b2FixtureDef)
from Box2D import (b2Vec2)

class ParallelWorldModel (Model):
    # Creates a copy of the given world by creating copies of all bodies
    def __init__(self, world):
        self.world = copyWorld(world)
        self.world.enableWarmStarting = True

    # Takes a step and prepares a dictionary with impulse-results for use by Predict
    def Step(self, world, timeStep, velocityIterations, positionIterations):
        self.world.Step(timeStep, velocityIterations, positionIterations)

        self.predictions = {}
        for c in self.world.contacts:
            res = []
            for p in c.manifold.points:
                res.append((p.id, p.normalImpulse, p.tangentImpulse))
            self.predictions[(c.fixtureA.body.userData, c.fixtureB.body.userData)] = res

    # Predict a result by looking up in the dictionary
    def Predict(self, contact):
        idA = contact.fixtureA.body.userData
        idB = contact.fixtureB.body.userData
        res1 = self.predictions.get((idA, idB), [])
        res2 = self.predictions.get((idB, idA), [])
        return res1+res2
