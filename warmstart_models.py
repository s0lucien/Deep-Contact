import random
import pandas as pd

from Box2D import b2ContactListener, b2World, b2FixtureDef, b2Vec2

import numpy as np


# Utility class used for gathering contact values
class CustomContactListener(b2ContactListener):
    def __init__(self):
        super(CustomContactListener, self).__init__()

        self.reset()

    def reset(self):
        self.df_c = pd.DataFrame(columns = ["master", "slave", "px", "py", "nx", "ny", "ni", "ti"])
        self.n_c = 0

    def PreSolve(self, contact, _):
        contact.userData = self.n_c
        for i in range(contact.manifold.pointCount):
            worldPoint = contact.worldManifold.points[i]
            px = worldPoint[0]
            py = worldPoint[1]

            normal = contact.worldManifold.normal
            nx = normal[0]
            ny = normal[1]

            master = contact.fixtureA.body.userData.id
            slave = contact.fixtureB.body.userData.id

            self.df_c.loc[self.n_c] = [master, slave, px, py, nx, ny, 0, 0]
            self.n_c += 1

    def PostSolve(self, contact, impulse):
        n_c = contact.userData
        for i in range(contact.manifold.pointCount):
            normal = impulse.normalImpulses[i]
            tangent = impulse.tangentImpulses[i]

            self.df_c.loc[n_c+i].ni = normal
            self.df_c.loc[n_c+i].ti = tangent


# Creates a copy of the given world by creating copies of all bodies
def copyWorld(world):
    copy = b2World(gravity=world.gravity, doSleep=world.allowSleeping)

    copy.continuousPhysics  = world.continuousPhysics

    copy.velocityThreshold = world.velocityThreshold
    copy.positionThreshold = world.positionThreshold

    for body in world.bodies:
        fixtures = []
        for fixture in body.fixtures:
            fixtures.append(b2FixtureDef(
                shape=fixture.shape,
                density=fixture.density,
                restitution=fixture.restitution,
                friction=fixture.friction
            ))

        copy.CreateBody(
            type=body.type,
            fixtures=fixtures,
            userData=body.userData,
            position=b2Vec2(body.position.x, body.position.y),
            angle=body.angle,
            linearVelocity=b2Vec2(body.linearVelocity.x, body.linearVelocity.y),
            angularVelocity=body.angularVelocity
        )

    for body in copy.bodies:
        body.sleepingAllowed = False


    return copy


# Stores normal and tangential impulse predictions for a contact in a dictionary
# The fact that there might be more than one contact between bodies makes this
# a bit more complicated, it is solved by also using coordinates
def storePredictions(predictionDict, contact, predictions):
    idA = contact.fixtureA.body.userData.id
    idB = contact.fixtureB.body.userData.id
    key = (idA, idB)

    if key in predictionDict:
        storedPredictions = predictionDict[key]
    else:
        storedPredictions = []
        predictionDict[key] = storedPredictions

    for i in range(len(predictions)):
        worldPoint = contact.worldManifold.points[i]
        px = worldPoint[0]
        py = worldPoint[1]

        ni = predictions[i][0]
        ti = predictions[i][1]

        storedPredictions.append([px, py, ni, ti])


# Stores the normal and tangential impulse prediction for a contact from a dictionary
# The fact that there might be more than one contact between bodies makes this
# a bit more complicated, it is solved by also using coordinates
def getPredictions(predictionDict, contact):
    idA = contact.fixtureA.body.userData.id
    idB = contact.fixtureB.body.userData.id
    key = (idA, idB)

    if key in predictionDict:
        storedPredictions = predictionDict[(idA, idB)]
    else:
        print("Contact not found in prediction dict")
        return [[0,0], [0,0]]

    predictions = []
    for i in range(contact.manifold.pointCount):
        worldPoint = contact.worldManifold.points[i]
        px = worldPoint[0]
        py = worldPoint[1]

        # We choose the predictions whose position is closest to the position of the contact point
        # Might be bad in case of extreme behaviour like very high velocities
        pred = min(storedPredictions, key=lambda p: (px-p[0])**2 + (py-p[1])**2)

        predictions.append(pred[2:4])

    return predictions



# A model has three functions - __init__, Step and PreSolve.
# __init__ takes different input for different models, depending on what they need
# Step and PreSolve takes the same input for all models, irregardless of whether a
# specific model needs the input, in order to make switching models easy
# NOTE: This is not actually a model intended to be used, simply an example and a
# way to unify some behaviour across other models
class Model(b2ContactListener):
    # Initializes the Model
    def __init__(self):
        super(Model, self).__init__()

    # Tells the model to take a step, and typically create and store predictions
    def Step(self, world, timeStep, velocityIterations, positionIterations):
        self.predictionDict = {}
        self.normalPairs = []
        self.tangentPairs = []

    # Takes a contact and sets the contact manifold points' normalImpulse and
    # tangentImpulse values, typically using the set predictions
    # Third argument is unused
    def PreSolve(self, contact, _):
        predictions = getPredictions(self.predictionDict, contact)

        for i in range(contact.manifold.pointCount):
            point = contact.manifold.points[i]
            point.normalImpulse = predictions[i][0]
            point.tangentImpulse = predictions[i][1]

    # Store impulse predictions and results for error calculations
    def PostSolve(self, contact, impulse):
        predictions = getPredictions(self.predictionDict, contact)

        for i in range(contact.manifold.pointCount):
            pred_ni = predictions[i][0]
            pred_ti = predictions[i][1]

            res_ni = impulse.normalImpulses[i]
            res_ti = impulse.tangentImpulses[i]

            self.normalPairs.append((pred_ni, res_ni))
            self.tangentPairs.append((pred_ti, res_ti))


# A model which effectively disables warm-starting, by using 0's as starting iterates
class NoWarmStartModel(Model):
    def __init__(self):
        super(NoWarmStartModel, self).__init__()

    def Step(self, world, timeStep, velocityIterations, positionIterations):
        super(NoWarmStartModel, self).Step(
            world, timeStep, velocityIterations, positionIterations
        )

    # Predicts 0's
    def PreSolve(self, contact, _):
        for i in range(contact.manifold.pointCount):
            point = contact.manifold.points[i]
            point.normalImpulse = 0
            point.tangentImpulse = 0

    # Store impulse predictions and results for error calculations
    def PostSolve(self, contact, impulse):
        for i in range(contact.manifold.pointCount):
            ni = impulse.normalImpulses[i]
            ti = impulse.tangentImpulses[i]

            self.normalPairs.append((0, ni))
            self.tangentPairs.append((0, ti))



# A model which does nothing, resulting in the simulator using the built-in warm-starting
class BuiltinWarmStartModel(Model):
    def __init__(self):
        super(BuiltinWarmStartModel, self).__init__()

    def Step(self, world, timeStep, velocityIterations, positionIterations):
        super(BuiltinWarmStartModel, self).Step(
            world, timeStep, velocityIterations, positionIterations
        )

    # Uses last steps results as predictions
    def PreSolve(self, contact, _):
        predictions = []
        for p in contact.manifold.points:
            predictions.append([p.normalImpulse, p.tangentImpulse])

        storePredictions(self.predictionDict, contact, predictions)

    # Store impulse predictions and results for error calculations
    def PostSolve(self, contact, impulse):
        super(BuiltinWarmStartModel, self).PostSolve(contact, impulse)



# Provides a very bad prediction, irregardless of input
class BadModel(Model):
    def __init__(self):
        super(BadModel, self).__init__()
        self.p = 50

    def Step(self, world, timeStep, velocityIterations, positionIterations):
        super(BadModel, self).Step(
            world, timeStep, velocityIterations, positionIterations
        )

    def PreSolve(self, contact, _):
        for point in contact.manifold.points:
            point.normalImpulse = self.p
            point.tangentImpulse = self.p

    # Store impulse predictions and results for error calculations
    def PostSolve(self, contact, impulse):
        for i in range(contact.manifold.pointCount):
            ni = impulse.normalImpulses[i]
            ti = impulse.tangentImpulses[i]

            self.normalPairs.append((self.p, ni))
            self.tangentPairs.append((self.p, ti))



# A model which predicts a random, some-what 'reasonable' set of impulses
class RandomModel(Model):
    # We manually choose a seed to ensure the same 'random' numbers each time
    def __init__(self, seed):
        super(RandomModel, self).__init__()

        random.seed(seed)

    def Step(self, world, timeStep, velocityIterations, positionIterations):
        super(RandomModel, self).Step(
            world, timeStep, velocityIterations, positionIterations
        )

    def PreSolve(self, contact, _):
        predictions = []
        for point in contact.manifold.points:
            # Normal impulses seems to be in the range 0 to 5
            point.normalImpulse = random.uniform(0, 5)

            # Tangential impulses seems to be in the range -2 to 2
            point.tangentImpulse = random.uniform(-2, 2)

            predictions.append([point.normalImpulse, point.tangentImpulse])

        storePredictions(self.predictionDict, contact, predictions)

    # Store impulse predictions and results for error calculations
    def PostSolve(self, contact, impulse):
        super(RandomModel, self).PostSolve(contact, impulse)



# A model which creates a copy of the current world, asks that copy to take a step,
# and reports the results back to the original world rounded to set accuracy
class CopyWorldModel(Model):
    # 'accuracy' is the argument passed to round, i.e. the number of decimals to round up to
    # if not set, no rounding is done
    def __init__(self, accuracy=None):
        super(CopyWorldModel, self).__init__()

        self.accuracy = accuracy

    # Creates a copy of the world, tells it to take a step
    # and prepares a dictionary with impulse-results for use by Predict
    def Step(self, world, timeStep, velocityIterations, positionIterations):
        super(CopyWorldModel, self).Step(
            world, timeStep, velocityIterations, positionIterations
        )

        copy = copyWorld(world)
        copy.Step(timeStep, velocityIterations, positionIterations)

        for contact in copy.contacts:
            res = []
            for p in contact.manifold.points:
                if self.accuracy != None:
                    normal = round(p.normalImpulse, self.accuracy)
                    tangent = round(p.tangentImpulse, self.accuracy)
                else:
                    normal = p.normalImpulse
                    tangent = p.tangentImpulse

                res.append([normal, tangent])

            storePredictions(self.predictionDict, contact, res)

    # Predict a result by looking up in dictionary
    def PreSolve(self, contact, old):
        super(CopyWorldModel, self).PreSolve(contact, old)

    # Store impulse predictions and results for error calculations
    def PostSolve(self, contact, impulse):
        super(CopyWorldModel, self).PostSolve(contact, impulse)

from sph_grid import Grid, dataframe_to_grid
# A model which takes the impulses from last step, similar to how the build-in
# warm start works, but then transfers them onto a grid, from the grid back
# to the particles, and then uses the results as predictions
class IdentityGridModel(Model):
    def __init__(self, p_ll, p_ur, xRes, yRes, h):
        super(IdentityGridModel, self).__init__()

        # Initialize the grid
        self.G = Grid(p_ll,p_ur,(xRes,yRes))
        self.h = h
        self.contact_channels = ['ni','ti']

        # Create custom contact listener
        self.ContactListener = CustomContactListener()

    def Step(self, world, timeStep, velocityIterations, positionIterations):
        super(IdentityGridModel, self).Step(
            world, timeStep, velocityIterations, positionIterations
        )

        # Reset
        self.ContactListener.reset()

        # Create a copy of world and step
        copy = copyWorld(world)
        copy.contactListener = self.ContactListener

        copy.Step(timeStep, velocityIterations, positionIterations)

        df_c = self.ContactListener.df_c
        N = df_c.shape[0]
        if N == 0:
            return

        # Transfer from particles to grids
        c_grids={}
        for c in self.contact_channels:
            c_grids[c]=dataframe_to_grid(G=self.G,channel=c,df=df_c,support_radius=self.h)
            c_grids[c]=np.flip(c_grids[c])

        # Extract contact positions
        c_pos =[(i,j) for i,j in zip(df_c.px.values,df_c.py.values)]

        # Transfer from grids to particles
        c_preds={}
        for c in self.contact_channels:
            c_preds[c]=self.G.collect(c_grids[c],c_pos)


        # Store predictions
        for i in range(N):
            row = df_c.loc[i]
            master = int(row['master'])
            slave = int(row['slave'])
            key = (master, slave)

            px = row['px']
            py = row['py']

#             import pdb; pdb.set_trace()
            prediction = [px, py] + [c_preds['ni'][i],c_preds['ti'][i]]
            
            if key in self.predictionDict:
                self.predictionDict[key].append(prediction)
            else:
                self.predictionDict[key] = [prediction]

    # Set impulses based on stored predictions
    def PreSolve(self, contact, old):
        super(IdentityGridModel, self).PreSolve(contact, old)

    # Store impulse predictions and results for error calculations
    def PostSolve(self, contact, impulse):
        super(IdentityGridModel, self).PostSolve(contact, impulse)