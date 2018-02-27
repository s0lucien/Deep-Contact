from model import Model
from util import copyWorld

from Box2D import (b2World)
from Box2D import (b2FixtureDef)
from Box2D import (b2Vec2)

class ParallelWorldModel (Model):
    def __init__(self, world):
        self.world = copyWorld(world)

    # Takes a step and prepares a dictionary with impulse-results for use by Predict
    def Step(self, world, timeStep, velocityIterations, positionIterations):
        self.world.Step(timeStep, velocityIterations, positionIterations)

        self.predictions = {}
        for c in self.world.contacts:
            if c.touching:
                res = []
                for p in c.manifold.points:
                    res.append((p.id, p.normalImpulse, p.tangentImpulse))
                self.predictions[(c.fixtureA.body.userData, c.fixtureB.body.userData)] = res

    # Predict a result by looking up in dictionary
    def Predict(self, contact):
        return self.predictions.get((contact.fixtureA.body.userData, contact.fixtureB.body.userData), [])
